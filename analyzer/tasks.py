import uuid
from concurrent.futures import ThreadPoolExecutor
from news.news import get_scrappers
from analyzer.zero import ZeroShotClassifier

tasks = {}
scrappers = get_scrappers()
classifier = ZeroShotClassifier()

def create_task(data):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'Pending',
        'input_data': data,
        'result': None,
        'error': None
    }
    # Start the task asynchronously
    process_task_async(task_id)
    return task_id

def process_task_async(task_id):
    task_input_data = tasks[task_id]['input_data']
    executor = ThreadPoolExecutor(max_workers=2)
    future = executor.submit(process_task, task_id, task_input_data)

    future.add_done_callback(lambda x: tasks[task_id].update({'status': 'Complete'}))

def process_task(task_id, task_data):
    task = tasks.get(task_id, None)
    try:
        news_list = scrappers['tass'].get_news()
        
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
                probabilities = classifier.predict(news, [pair['class1'], pair['class2']])

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


def get_task_status(task_id):
    task = tasks.get(task_id, None)
    if task:
        return {'status': task['status'], 'result': task['result'], 'error': task['error']}
    return {'status': 'Task not found'}