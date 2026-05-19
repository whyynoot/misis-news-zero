import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from analyzer.constants import parse_bool
from analyzer.zero import ZeroShotClassifier
from news.sources import collect_news_entries

tasks = {}
executor = ThreadPoolExecutor(max_workers=2)


@lru_cache(maxsize=1)
def get_classifier():
    return ZeroShotClassifier()

def create_task(data):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'Pending',
        'kind': 'analysis',
        'input_data': data,
        'result': None,
        'error': None
    }
    # Start the task asynchronously
    process_task_async(task_id)
    return task_id


def create_monitoring_task(data):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'Pending',
        'kind': 'monitoring',
        'input_data': data,
        'result': None,
        'error': None
    }
    process_monitoring_task_async(task_id)
    return task_id


def create_history_backfill_task(data):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'Pending',
        'kind': 'history_backfill',
        'input_data': data,
        'result': None,
        'error': None
    }
    process_history_backfill_task_async(task_id)
    return task_id


def process_task_async(task_id):
    task_input_data = tasks[task_id]['input_data']
    tasks[task_id]['status'] = 'Running'
    future = executor.submit(process_task, task_id, task_input_data)
    future.add_done_callback(lambda x: finalize_task(task_id, x))


def process_monitoring_task_async(task_id):
    task_input_data = tasks[task_id]['input_data']
    tasks[task_id]['status'] = 'Running'
    future = executor.submit(process_monitoring_task, task_id, task_input_data)
    future.add_done_callback(lambda x: finalize_task(task_id, x))


def process_history_backfill_task_async(task_id):
    task_input_data = tasks[task_id]['input_data']
    tasks[task_id]['status'] = 'Running'
    future = executor.submit(process_history_backfill_task, task_id, task_input_data)
    future.add_done_callback(lambda x: finalize_task(task_id, x))


def finalize_task(task_id, future):
    task = tasks.get(task_id)
    if not task:
        return

    try:
        future.result()
    except Exception as exc:
        task['error'] = str(exc)

    task['status'] = 'Failed' if task['error'] else 'Complete'

def process_task(task_id, task_data):
    task = tasks.get(task_id, None)
    if task:
        task['status'] = 'Running'
    try:
        news_list = [entry.get("text", "") for entry in collect_news_entries() if entry.get("text")]
        
        if len(news_list) == 0:
            task['error'] = "Parsing error"
            return

        # Extract pairs of classes from the input task data
        pairs = task_data.get('pairs', [])

        # Initialize the results and summary
        results = []
        summary = {f"{pair['class1']} / {pair['class2']}": [0, 0, 0] for pair in pairs}

        # Process each news item and classify based on the pairs
        for news in news_list:
            news_result = {
                'text': news,
                'classification': []
            }

            # Classify results for each pair
            for pair in pairs:
                probabilities = get_classifier().predict(news, [pair['class1'], pair['class2']])

                classification_result = {
                    f"{pair['class1']} / {pair['class2']}": probabilities
                }

                # Update the summary for this pair
                summary[f"{pair['class1']} / {pair['class2']}"][0] += probabilities[0]
                summary[f"{pair['class1']} / {pair['class2']}"][1] += probabilities[1]
                summary[f"{pair['class1']} / {pair['class2']}"][2] += 1

                news_result['classification'].append(classification_result)

            results.append(news_result)

        # Calculate averages for the summary
        for key, value in summary.items():
            value[0] = value[0] / value[2] if value[2] != 0 else 0
            value[1] = value[1] / value[2] if value[2] != 0 else 0

        # Store the results and summary
        if task and len(results) > 0:
            task['result'] = {
                'news_results': results,
                'summary': summary
            }
        else:
            task['error'] = "Unexpected error"
    except Exception as e:
        task['error'] = str(e)


def process_monitoring_task(task_id, task_data):
    task = tasks.get(task_id, None)
    if task:
        task['status'] = 'Running'
    try:
        from analyzer.monitoring_service import run_monitoring_pipeline

        result = run_monitoring_pipeline(
            engine=task_data.get('engine') or 'bert',
            force=parse_bool(task_data.get('force'), False),
            summarize=parse_bool(task_data.get('summarize'), False),
            period=task_data.get('period') or 'day',
            bootstrap_if_empty=parse_bool(task_data.get('bootstrap_if_empty'), False),
            limit=task_data.get('limit'),
        )
        if task:
            task['result'] = result
    except Exception as e:
        if task:
            task['error'] = str(e)


def process_history_backfill_task(task_id, task_data):
    task = tasks.get(task_id, None)
    if task:
        task['status'] = 'Running'
    try:
        from analyzer.monitoring_service import backfill_news_history

        result = backfill_news_history(
            start_date=task_data.get('start_date'),
            end_date=task_data.get('end_date'),
            days=task_data.get('days'),
            source_names=task_data.get('sources'),
            limit_per_day=task_data.get('limit_per_day'),
            total_limit=task_data.get('limit'),
            delay_seconds=task_data.get('delay_seconds') or 0.0,
            engine=task_data.get('engine') or 'bert',
            force=parse_bool(task_data.get('force'), False),
            summarize=parse_bool(task_data.get('summarize'), False),
            period=task_data.get('period') or 'day',
        )
        if task:
            task['result'] = result
    except Exception as e:
        if task:
            task['error'] = str(e)


def get_task_status(task_id):
    task = tasks.get(task_id, None)
    if task:
        return {'status': task['status'], 'kind': task.get('kind'), 'result': task['result'], 'error': task['error']}
    return {'status': 'Task not found'}
