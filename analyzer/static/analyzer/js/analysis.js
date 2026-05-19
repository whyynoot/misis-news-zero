document.addEventListener('DOMContentLoaded', () => {
    const addClassButton = document.getElementById('add-class');
    const pairsContainer = document.getElementById('classification-pairs');
    const analyzeButton = document.getElementById('analyze');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const analyzeLabel = document.getElementById('analyze-label');
    const resetButton = document.getElementById('reset-pairs');
    const clearResultsButton = document.getElementById('clear-results');
    const loadingIndicator = document.getElementById('loading');
    const resultsBody = document.getElementById('results-body');
    const summaryBody = document.getElementById('summary-body');
    const resultsHeader = document.getElementById('results-header');
    const resultsMeta = document.getElementById('results-meta');
    const defaultPairsNode = document.getElementById('default-pairs-data');

    const NEUTRAL_THRESHOLD = 0.8;
    const defaultPairs = JSON.parse(defaultPairsNode?.textContent || '[]');

    function setAnalyzing(isRunning) {
        analyzeButton.disabled = isRunning;
        analyzeSpinner.classList.toggle('d-none', !isRunning);
        analyzeLabel.textContent = isRunning ? 'Обработка...' : 'Запустить анализ';
    }

    function renderTile(label, value, note = '') {
        const article = document.createElement('article');
        article.className = 'summary-tile';

        const labelNode = document.createElement('div');
        labelNode.className = 'summary-label';
        labelNode.textContent = label;

        const valueNode = document.createElement('div');
        valueNode.className = 'summary-value';
        valueNode.textContent = value;

        article.append(labelNode, valueNode);

        if (note) {
            const noteNode = document.createElement('div');
            noteNode.className = 'summary-note';
            noteNode.textContent = note;
            article.appendChild(noteNode);
        }

        return article;
    }

    function renderMetaPlaceholder() {
        resultsMeta.replaceChildren(
            renderTile('Новостей', '--'),
            renderTile('Пар', '--'),
            renderTile('Нейтрально', '--'),
            renderTile('Уверенно', '--')
        );
    }

    function createInput(placeholder, value) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'input';
        input.placeholder = placeholder;
        input.value = value || '';
        return input;
    }

    function renderPairRow(pair = { class1: '', class2: '' }) {
        const wrapper = document.createElement('div');
        wrapper.className = 'pair-row';

        const firstField = document.createElement('label');
        firstField.className = 'pair-field';
        const firstIndex = document.createElement('span');
        firstIndex.className = 'pair-index';
        firstIndex.textContent = '1';
        firstField.append(firstIndex, createInput('Первая формулировка', pair.class1));

        const secondField = document.createElement('label');
        secondField.className = 'pair-field';
        const secondIndex = document.createElement('span');
        secondIndex.className = 'pair-index';
        secondIndex.textContent = '2';
        secondField.append(secondIndex, createInput('Вторая формулировка', pair.class2));

        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'icon-button';
        removeButton.title = 'Удалить пару';
        removeButton.setAttribute('aria-label', 'Удалить пару');
        removeButton.textContent = '×';
        removeButton.addEventListener('click', () => wrapper.remove());

        wrapper.append(firstField, secondField, removeButton);
        pairsContainer.appendChild(wrapper);
    }

    function resetPairs() {
        pairsContainer.replaceChildren();
        if (defaultPairs.length === 0) {
            renderPairRow();
            return;
        }
        defaultPairs.forEach(renderPairRow);
    }

    function setSingleCell(tbody, text, colspan = 1, className = 'text-muted') {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = colspan;
        cell.className = className;
        cell.textContent = text;
        row.appendChild(cell);
        tbody.replaceChildren(row);
    }

    function clearResults() {
        const headerCell = document.createElement('th');
        headerCell.textContent = 'Текст новости';
        resultsHeader.replaceChildren(headerCell);
        setSingleCell(resultsBody, 'Запустите анализ, чтобы увидеть решения.');
        setSingleCell(summaryBody, 'Нет запусков.', 3);
        renderMetaPlaceholder();
    }

    function collectPairs() {
        const pairs = [];
        pairsContainer.querySelectorAll('.pair-row').forEach((row) => {
            const inputs = row.querySelectorAll('input');
            const class1 = inputs[0]?.value.trim();
            const class2 = inputs[1]?.value.trim();
            if (class1 && class2) {
                pairs.push({ class1, class2 });
            }
        });
        return pairs;
    }

    function populateSummary(summary) {
        summaryBody.replaceChildren();
        if (!summary || Object.keys(summary).length === 0) {
            setSingleCell(summaryBody, 'Нет агрегированных результатов.', 3);
            return;
        }

        Object.entries(summary).forEach(([key, value]) => {
            const row = document.createElement('tr');
            const pairCell = document.createElement('td');
            const firstCell = document.createElement('td');
            const secondCell = document.createElement('td');

            pairCell.textContent = key;
            firstCell.textContent = Number(value[0] ?? 0).toFixed(2);
            secondCell.textContent = Number(value[1] ?? 0).toFixed(2);
            row.append(pairCell, firstCell, secondCell);
            summaryBody.appendChild(row);
        });
    }

    function updateMeta(results, pairs) {
        const pairCount = pairs.length;
        const totalItems = results.length;
        const decisions = pairCount * totalItems;
        let neutralCount = 0;
        let confidentCount = 0;

        results.forEach((newsItem) => {
            newsItem.classification.forEach((classification) => {
                const [[, probabilities]] = Object.entries(classification);
                const [prob1, prob2] = probabilities;
                if (Math.max(prob1, prob2) >= NEUTRAL_THRESHOLD) {
                    confidentCount += 1;
                } else {
                    neutralCount += 1;
                }
            });
        });

        const neutralShare = decisions ? Math.round((neutralCount / decisions) * 100) : 0;
        const confidentShare = decisions ? Math.round((confidentCount / decisions) * 100) : 0;
        const stamp = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });

        resultsMeta.replaceChildren(
            renderTile('Новостей', totalItems || '--', 'обработано'),
            renderTile('Пар', pairCount || '--', 'сравнений на новость'),
            renderTile('Нейтрально', `${neutralShare}%`, 'ниже порога'),
            renderTile('Уверенно', `${confidentShare}%`, `обновлено ${stamp}`)
        );
    }

    function populateResults(results, pairs) {
        loadingIndicator.classList.add('d-none');
        setAnalyzing(false);

        const headerCell = document.createElement('th');
        headerCell.textContent = 'Текст новости';
        resultsHeader.replaceChildren(headerCell);
        pairs.forEach((pair) => {
            const cell = document.createElement('th');
            cell.textContent = `${pair.class1} / ${pair.class2}`;
            resultsHeader.appendChild(cell);
        });

        resultsBody.replaceChildren();
        if (!Array.isArray(results) || results.length === 0) {
            setSingleCell(resultsBody, 'Нет новостей в ответе задачи.', pairs.length + 1);
            renderMetaPlaceholder();
            return;
        }

        results.forEach((newsItem) => {
            const row = document.createElement('tr');
            const textCell = document.createElement('td');
            const textWrap = document.createElement('div');
            textWrap.className = 'truncate-text';
            textWrap.title = newsItem.text || '';
            textWrap.textContent = newsItem.text || '';
            textCell.appendChild(textWrap);
            row.appendChild(textCell);

            newsItem.classification.forEach((classification, index) => {
                const [[, probabilities]] = Object.entries(classification);
                const [prob1, prob2] = probabilities;
                let result = 'Нейтрально';
                if (prob1 >= NEUTRAL_THRESHOLD && prob1 >= prob2) {
                    result = pairs[index]?.class1 || 'первая формулировка';
                } else if (prob2 >= NEUTRAL_THRESHOLD && prob2 > prob1) {
                    result = pairs[index]?.class2 || 'вторая формулировка';
                }

                const resultCell = document.createElement('td');
                resultCell.title = `${Number(prob1).toFixed(2)} / ${Number(prob2).toFixed(2)}`;
                resultCell.textContent = result;
                row.appendChild(resultCell);
            });

            resultsBody.appendChild(row);
        });

        updateMeta(results, pairs);
    }

    function showTaskError(message) {
        loadingIndicator.classList.add('d-none');
        setAnalyzing(false);
        setSingleCell(resultsBody, message, 1, 'text-danger');
    }

    function pollTaskStatus(taskId, pairs, delay = 2000) {
        const intervalId = setInterval(() => {
            fetch(`/api/task/${taskId}/`)
                .then((response) => response.json())
                .then((data) => {
                    if (data.status === 'Complete') {
                        clearInterval(intervalId);
                        if (data.error || !data.result) {
                            showTaskError(`Ошибка задачи: ${data.error || 'задача завершилась без результата'}`);
                            return;
                        }
                        populateResults(data.result.news_results, pairs);
                        populateSummary(data.result.summary);
                    } else if (data.status === 'Failed') {
                        clearInterval(intervalId);
                        showTaskError(`Ошибка задачи: ${data.error || 'неизвестная ошибка'}`);
                    }
                })
                .catch((error) => {
                    clearInterval(intervalId);
                    showTaskError(`Ошибка задачи: ${error}`);
                });
        }, delay);
    }

    addClassButton.addEventListener('click', () => renderPairRow());
    resetButton.addEventListener('click', resetPairs);
    clearResultsButton.addEventListener('click', clearResults);
    analyzeButton.addEventListener('click', () => {
        loadingIndicator.classList.remove('d-none');
        setAnalyzing(true);

        const pairs = collectPairs();
        if (pairs.length === 0) {
            showTaskError('Добавьте хотя бы одну пару, чтобы запустить анализ.');
            return;
        }

        fetch('/api/task/', {
            method: 'POST',
            body: JSON.stringify({ pairs }),
            headers: { 'Content-Type': 'application/json' },
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.task_id) {
                    pollTaskStatus(data.task_id, pairs);
                } else {
                    showTaskError('Не удалось создать задачу.');
                }
            })
            .catch((error) => showTaskError(`Ошибка запроса: ${error}`));
    });

    resetPairs();
    renderMetaPlaceholder();
});
