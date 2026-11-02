#!/usr/bin/env python3
"""Qualify constraints on an already running TP2 pair, once in AR and MTP.

Use an isolated pair with at least two sessions and thinking disabled. Responses,
stream chunks and per-request timings are retained even when a check fails.
In MTP mode every request must execute actual drafts, constrained ones too:
rank 1 rebuilds their constraint and mirrors their multi-token cycles.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8100')
parser.add_argument('--model', default='qwen3.8-flash-next')
parser.add_argument('--speculative', choices=('off', 'mtp'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = {'url': args.url, 'model': args.model,
          'expected_speculative': args.speculative, 'requests': {}, 'passed': False}
args_schema = {
    'type': 'object', 'properties': {
        'city': {'type': 'string', 'enum': ['Berlin']},
        'units': {'type': 'string', 'enum': ['celsius']}},
    'required': ['city', 'units'], 'additionalProperties': False}
schema = {
    'type': 'object', 'properties': {
        'city': {'type': 'string', 'enum': ['Berlin']},
        'temperature': {'type': 'integer', 'enum': [21]},
        'ok': {'type': 'boolean', 'enum': [True]}},
    'required': ['city', 'temperature', 'ok'], 'additionalProperties': False}
common = {'model': args.model, 'temperature': 0, 'seed': 31, 'max_tokens': 128,
          'presence_penalty': 0, 'frequency_penalty': 0, 'repeat_penalty': 1,
          'chat_template_kwargs': {'enable_thinking': False}}
tool = {
    **common, 'messages': [{'role': 'user', 'content':
        'Call lookup_city to look up Berlin in celsius. Use the tool, do not answer directly.'}],
    'tools': [{'type': 'function', 'function': {
        'name': 'lookup_city', 'description': 'Look up a city temperature.',
        'parameters': args_schema, 'strict': True}}], 'parallel_tool_calls': False}
structured = {
    **common, 'messages': [{'role': 'user', 'content':
        'Return Berlin, temperature 21, and ok true as JSON.'}],
    'response_format': {'type': 'json_schema', 'json_schema': {
        'name': 'weather', 'strict': True, 'schema': schema}}}
plain = {**common, 'messages': [{'role': 'user', 'content':
         'Count from one to twenty in English words.'}]}


def ask(name, body):
    start = time.perf_counter()
    req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                 json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    raw = []
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            status = response.status
            if body.get('stream'):
                text, calls, usage, finish = '', {}, {}, None
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        break
                    chunk = json.loads(data)
                    raw.append(chunk)
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        finish = choice.get('finish_reason') or finish
                        delta = choice.get('delta', {})
                        text += delta.get('content') or ''
                        for item in delta.get('tool_calls') or []:
                            call = calls.setdefault(item.get('index', 0),
                                {'function': {'name': '', 'arguments': ''}})
                            function = item.get('function', {})
                            call['function']['name'] += function.get('name') or ''
                            call['function']['arguments'] += function.get('arguments') or ''
                result = {'choices': [{'message': {
                    'content': text, 'tool_calls': list(calls.values())},
                    'finish_reason': finish}], 'usage': usage}
            else:
                result = json.load(response)
        wall = time.perf_counter() - start
        record = {'body': body, 'status': status, 'wall_seconds': wall, 'response': result}
        if raw:
            record['chunks'] = raw
        report['requests'][name] = record
        assert status == 200, record
        choice = result['choices'][0]
        message, usage = choice['message'], result['usage']
        if body.get('tools'):
            assert choice['finish_reason'] == 'tool_calls', record
            calls = message['tool_calls']
            assert len(calls) == 1, record
            assert calls[0]['function']['name'] == 'lookup_city', record
            assert json.loads(calls[0]['function']['arguments']) == {
                'city': 'Berlin', 'units': 'celsius'}, record
        elif body.get('response_format'):
            assert choice['finish_reason'] == 'stop', record
            value = json.loads(message['content'])
            assert value == {'city': 'Berlin', 'temperature': 21, 'ok': True}, record
            assert type(value['temperature']) is int and type(value['ok']) is bool, record
        else:
            assert message['content'] and not message.get('tool_calls'), record
        if args.speculative == 'mtp':
            assert usage['draft_tokens'] > 0, record
        if args.speculative == 'off':
            assert usage['draft_tokens'] == 0, record
        for key in ('prefill_ms', 'decode_ms', 'queue_ms', 'ttft_ms', 'cache_restore_ms'):
            assert key in usage['gufo'], (key, record)
        print(f'{name}: HTTP {status}, finish={choice["finish_reason"]}, '
              f'drafts={usage["draft_tokens"]}, wall={wall:.3f}s', flush=True)
        return record
    except urllib.error.HTTPError as e:
        report['requests'][name] = {'body': body, 'status': e.code,
                                    'wall_seconds': time.perf_counter() - start,
                                    'error': e.read().decode()}
        raise
    except Exception as e:
        record = report['requests'].setdefault(name, {'body': body,
            'wall_seconds': time.perf_counter() - start})
        record['error'] = str(e)
        raise


try:
    ask('tool_auto', tool)
    ask('tool_required', {**tool, 'tool_choice': 'required'})
    ask('tool_stream', {**tool, 'stream': True,
                        'stream_options': {'include_usage': True}})
    first = ask('schema', structured)
    streamed = ask('schema_stream', {**structured, 'stream': True,
                                    'stream_options': {'include_usage': True}})
    assert (first['response']['choices'][0]['message']['content'] ==
            streamed['response']['choices'][0]['message']['content'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(ask, 'parallel_tool', tool),
                pool.submit(ask, 'parallel_schema', structured)]
        for job in jobs:
            job.result()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(ask, 'mixed_schema', structured),
                pool.submit(ask, 'mixed_plain', plain)]
        for job in jobs:
            job.result()
    report['passed'] = True
finally:
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
