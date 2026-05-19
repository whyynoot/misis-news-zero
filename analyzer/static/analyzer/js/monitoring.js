document.addEventListener('DOMContentLoaded', () => {
    const factorBody = document.getElementById('factor-body');
    const factorSelect = document.getElementById('factor-select');
    const factorDescription = document.getElementById('factor-description');
    const factorHistoryBody = document.getElementById('factor-history-body');
    const newsBody = document.getElementById('news-body');
    const spikeBody = document.getElementById('spike-body');
    const statSentiment = document.getElementById('stat-sentiment');
    const statRisk = document.getElementById('stat-risk');
    const statNews = document.getElementById('stat-news');
    const statWindow = document.getElementById('stat-window');
    const statRun = document.getElementById('stat-run');
    const statRunMeta = document.getElementById('stat-run-meta');
    const factorLatest = document.getElementById('factor-latest');
    const factorDelta = document.getElementById('factor-delta');
    const factorSignal = document.getElementById('factor-signal');
    const factorBalance = document.getElementById('factor-balance');
    const refreshBtn = document.getElementById('refresh-now');
    const historyBackfillBtn = document.getElementById('history-backfill');
    const reloadBtn = document.getElementById('reload-data');
    const engineSelect = document.getElementById('engine-select');
    const actionStatus = document.getElementById('action-status');
    const summaryTitle = document.getElementById('summary-title');
    const summaryText = document.getElementById('summary-text');
    const summaryTrend = document.getElementById('summary-trend');
    const summaryRisk = document.getElementById('summary-risk');
    const summaryConfidence = document.getElementById('summary-confidence');
    const summaryDrivers = document.getElementById('summary-drivers');

    const STATUS_LABELS = {
        success: 'успешно',
        skipped: 'пропущено',
        failed: 'ошибка',
        partial_failed: 'частично',
    };

    const MODE_LABELS = {
        seed: 'начальная загрузка',
        backfill: 'историческая загрузка',
        update: 'обновление',
    };

    let overviewChart;
    let factorChart;
    let selectedFactorKey = '';
    let selectedEngine = engineSelect?.value || 'llm';
    let lastFeed = null;
    const monitoringWindowDays = 3650;

    function fmt(value, digits = 2) {
        const number = Number.parseFloat(value ?? 0);
        return Number.isFinite(number) ? number.toFixed(digits) : '--';
    }

    function signed(value, digits = 2) {
        const number = Number.parseFloat(value ?? 0);
        if (!Number.isFinite(number)) return '--';
        return `${number >= 0 ? '+' : ''}${number.toFixed(digits)}`;
    }

    function labelFromMap(map, value, fallback = '--') {
        if (!value) return fallback;
        return map[value] || value;
    }

    function sourceLabel(value) {
        const labels = {
            tass: 'ТАСС',
            interfax: 'Интерфакс',
            seed: 'начальный набор',
        };
        return labels[value] || value || '';
    }

    function engineLabel(value) {
        const labels = {
            bert: 'BERT',
            llm: 'LLM',
        };
        return labels[value] || value || 'LLM';
    }

    function formatDate(value) {
        if (!value) return '--';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });
    }

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    function chartTheme() {
        return {
            text: cssVar('--text') || '#eef2ef',
            muted: cssVar('--muted') || '#9fa9a3',
            grid: cssVar('--border') || '#303833',
            surface: cssVar('--surface-2') || '#1f2422',
            accent: cssVar('--accent') || '#2dd4bf',
            blue: cssVar('--blue') || '#60a5fa',
            amber: cssVar('--amber') || '#f7b955',
        };
    }

    function sentimentClass(value) {
        const number = Number.parseFloat(value ?? 0);
        if (number > 0.2) return 'sentiment-positive';
        if (number < -0.2) return 'sentiment-negative';
        return 'sentiment-neutral';
    }

    function setValueClass(node, baseClass, value) {
        node.className = `${baseClass} ${sentimentClass(value)}`;
    }

    function getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta?.content || '';
    }

    function setActionStatus(message, state = '') {
        if (!actionStatus) return;
        actionStatus.className = `action-status${state ? ` is-${state}` : ''}`;
        actionStatus.textContent = message;
    }

    function setActionButtonsDisabled(disabled) {
        [refreshBtn, historyBackfillBtn, reloadBtn].filter(Boolean).forEach((button) => {
            button.disabled = disabled;
        });
    }

    function clearTable(tbody, message, colspan, className = 'text-muted') {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = colspan;
        cell.className = className;
        cell.textContent = message;
        row.appendChild(cell);
        tbody.replaceChildren(row);
    }

    function createTextCell(text, className = '') {
        const cell = document.createElement('td');
        if (className) cell.className = className;
        cell.textContent = text;
        return cell;
    }

    function createNewsLink(news) {
        if (news?.url) {
            const link = document.createElement('a');
            link.href = news.url;
            link.target = '_blank';
            link.rel = 'noreferrer';
            link.textContent = news.title || 'Новость';
            return link;
        }
        const span = document.createElement('span');
        span.textContent = news?.title || 'Нет новости';
        return span;
    }

    function renderOverview(overview) {
        statSentiment.textContent = fmt(overview.average_sentiment);
        setValueClass(statSentiment, 'metric-value', overview.average_sentiment);
        statRisk.textContent = fmt(overview.risk_index);
        statNews.textContent = overview.window_news_count ?? 0;
        statWindow.textContent = `Движок: ${engineLabel(selectedEngine)}. Окно: ${monitoringWindowDays} дней. Последняя дата: ${overview.latest_date || '--'}`;

        const run = overview.last_run || {};
        statRun.textContent = labelFromMap(STATUS_LABELS, run.status);
        statRunMeta.textContent = run.finished_at
            ? `${engineLabel(run.engine || selectedEngine)} · ${labelFromMap(MODE_LABELS, run.mode, 'запуск')} · получено: ${run.news_fetched || 0} · сохранено: ${run.news_stored || 0}`
            : 'Запусков пока нет.';
    }

    function renderOverviewChart(timeline) {
        const ctx = document.getElementById('overviewChart').getContext('2d');
        const colors = chartTheme();
        if (overviewChart) overviewChart.destroy();

        overviewChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeline.map((item) => item.date),
                datasets: [
                    {
                        label: 'Средняя тональность',
                        data: timeline.map((item) => item.average_sentiment),
                        borderColor: colors.accent,
                        backgroundColor: 'rgba(45, 212, 191, 0.12)',
                        pointRadius: 2,
                        tension: 0.28,
                        borderWidth: 2,
                        fill: true,
                    },
                    {
                        label: 'Индекс давления',
                        data: timeline.map((item) => item.risk_index),
                        borderColor: colors.amber,
                        backgroundColor: 'rgba(247, 185, 85, 0.08)',
                        pointRadius: 2,
                        tension: 0.28,
                        borderWidth: 2,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.text, boxWidth: 10, boxHeight: 10 } },
                },
                scales: {
                    x: {
                        ticks: { color: colors.muted, autoSkip: true, maxTicksLimit: 10 },
                        grid: { color: colors.grid },
                    },
                    y: {
                        ticks: { color: colors.muted },
                        grid: { color: colors.grid },
                        suggestedMin: -1,
                        suggestedMax: 1,
                    },
                },
            },
        });
    }

    function renderFactorTable(factors) {
        factorBody.replaceChildren();
        if (!factors || factors.length === 0) {
            clearTable(factorBody, 'Нет данных по факторам.', 4);
            return;
        }

        factors.forEach((factor) => {
            const row = document.createElement('tr');

            const factorCell = document.createElement('td');
            const name = document.createElement('div');
            name.className = 'cell-main';
            name.textContent = factor.name;
            const description = document.createElement('div');
            description.className = 'cell-note';
            description.textContent = `Сигнал: ${factor.relevant_news_count}/${factor.total_news_count}. ${factor.description || ''}`;
            factorCell.append(name, description);

            const sentimentCell = createTextCell(fmt(factor.latest_sentiment), sentimentClass(factor.latest_sentiment));
            const deltaCell = createTextCell(signed(factor.delta_from_previous), sentimentClass(factor.delta_from_previous));

            const actionCell = document.createElement('td');
            actionCell.className = 'row-action';
            const button = document.createElement('button');
            button.className = 'button button-ghost button-small';
            button.type = 'button';
            button.dataset.factor = factor.key;
            button.textContent = 'Открыть';
            button.addEventListener('click', () => {
                selectedFactorKey = factor.key;
                factorSelect.value = selectedFactorKey;
                loadMonitoringData(selectedFactorKey);
            });
            actionCell.appendChild(button);

            row.append(factorCell, sentimentCell, deltaCell, actionCell);
            factorBody.appendChild(row);
        });
    }

    function syncFactorSelect(factors) {
        const previous = selectedFactorKey || factorSelect.value;
        factorSelect.replaceChildren();

        factors.forEach((factor) => {
            const option = document.createElement('option');
            option.value = factor.key;
            option.textContent = factor.name;
            factorSelect.appendChild(option);
        });

        if (factors.length > 0) {
            selectedFactorKey = factors.some((item) => item.key === previous) ? previous : factors[0].key;
            factorSelect.value = selectedFactorKey;
        }
    }

    function renderFactorChart(timeline, label) {
        const ctx = document.getElementById('factorChart').getContext('2d');
        const colors = chartTheme();
        if (factorChart) factorChart.destroy();

        factorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeline.map((item) => item.date),
                datasets: [
                    {
                        label: label || 'Фактор',
                        data: timeline.map((item) => item.average_sentiment),
                        borderColor: colors.blue,
                        backgroundColor: 'rgba(96, 165, 250, 0.12)',
                        pointRadius: 2,
                        tension: 0.28,
                        borderWidth: 2,
                        fill: true,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.text, boxWidth: 10, boxHeight: 10 } },
                },
                scales: {
                    x: {
                        ticks: { color: colors.muted, autoSkip: true, maxTicksLimit: 8 },
                        grid: { color: colors.grid },
                    },
                    y: {
                        ticks: { color: colors.muted },
                        grid: { color: colors.grid },
                        suggestedMin: -1,
                        suggestedMax: 1,
                    },
                },
            },
        });
    }

    function renderFactorDetail(detail) {
        factorDescription.textContent = detail.description || '--';

        const latest = detail.latest || {};
        factorLatest.textContent = fmt(latest.average_sentiment);
        setValueClass(factorLatest, 'summary-value', latest.average_sentiment);
        factorDelta.textContent = signed(latest.delta_from_previous);
        setValueClass(factorDelta, 'summary-value', latest.delta_from_previous);
        factorSignal.textContent = `${latest.relevant_news_count ?? 0}/${latest.total_news_count ?? 0}`;
        factorBalance.textContent = `${latest.positive_hits ?? 0} / ${latest.negative_hits ?? 0}`;

        const timeline = detail.timeline || [];
        factorHistoryBody.replaceChildren();
        if (timeline.length === 0) {
            clearTable(factorHistoryBody, 'История по фактору пока пуста.', 4);
        } else {
            [...timeline].reverse().slice(0, 12).forEach((item) => {
                const row = document.createElement('tr');
                row.append(
                    createTextCell(item.date),
                    createTextCell(fmt(item.average_sentiment), sentimentClass(item.average_sentiment)),
                    createTextCell(signed(item.delta_from_previous), sentimentClass(item.delta_from_previous)),
                    createTextCell(`${item.relevant_news_count}/${item.total_news_count}`)
                );
                factorHistoryBody.appendChild(row);
            });
        }

        renderFactorChart(timeline, detail.name);
        renderRecentNews(detail.recent_news || []);
    }

    function renderRecentNews(recentNews) {
        newsBody.replaceChildren();
        if (recentNews.length === 0) {
            clearTable(newsBody, 'Нет сигнальных новостей по выбранному фактору.', 3);
            return;
        }

        recentNews.forEach((item) => {
            const row = document.createElement('tr');
            const newsCell = document.createElement('td');
            const title = document.createElement('div');
            title.className = 'cell-main';
            title.appendChild(createNewsLink(item));
            const note = document.createElement('div');
            note.className = 'cell-note';
            note.textContent = item.summary || item.category || sourceLabel(item.source);
            newsCell.append(title, note);

            row.append(
                createTextCell(formatDate(item.published_at)),
                newsCell,
                createTextCell(fmt(item.sentiment_score), sentimentClass(item.sentiment_score))
            );
            newsBody.appendChild(row);
        });
    }

    function renderSpikes(spikes) {
        spikeBody.replaceChildren();
        if (!spikes || spikes.length === 0) {
            clearTable(spikeBody, 'Пока нет выраженных всплесков.', 4);
            return;
        }

        spikes.forEach((item) => {
            const news = item.spotlight_news;
            const newsCell = document.createElement('td');
            const title = document.createElement('div');
            title.className = 'cell-main';
            title.appendChild(createNewsLink(news));
            newsCell.appendChild(title);

            if (news) {
                const meta = document.createElement('div');
                meta.className = 'cell-note';
                meta.textContent = `${fmt(news.sentiment_score)} · ${sourceLabel(news.source)}`;
                newsCell.appendChild(meta);
            }

            const row = document.createElement('tr');
            row.append(
                createTextCell(item.date),
                createTextCell(item.factor_name),
                createTextCell(signed(item.delta_from_previous), sentimentClass(item.delta_from_previous)),
                newsCell
            );
            spikeBody.appendChild(row);
        });
    }

    function renderFeed(data) {
        lastFeed = data;
        selectedEngine = data.engine || selectedEngine;
        if (engineSelect) engineSelect.value = selectedEngine;
        renderOverview(data.overview || {});
        renderOverviewChart(data.timeline || []);
        syncFactorSelect(data.factors || []);
        renderFactorTable(data.factors || []);
        renderFactorDetail(data.factor_detail || {});
        renderSpikes(data.spikes || []);
    }

    function renderSummary(payload) {
        summaryDrivers.replaceChildren();
        summaryTrend.textContent = 'trend: --';
        summaryRisk.textContent = 'risk: --';
        summaryConfidence.textContent = 'confidence: --';

        if (selectedEngine !== 'llm') {
            summaryTitle.textContent = 'LLM-сводка';
            summaryText.textContent = 'Переключитесь на LLM, чтобы увидеть summary по выбранному фактору.';
            return;
        }

        const factors = payload.factors || [];
        const item = factors.find((factor) => factor.factor_id === selectedFactorKey) || factors[0];
        summaryTitle.textContent = item?.factor_name ? `Сводка: ${item.factor_name}` : 'LLM-сводка';
        if (!item?.summary && !payload?.llm_enabled) {
            summaryTitle.textContent = 'LLM недоступна';
            summaryText.textContent = 'LLM выключена в конфигурации или summary еще не сформировано.';
            return;
        }
        if (!item?.summary) {
            summaryText.textContent = 'Сводка пока не сформирована. Запустите LLM-обновление с summary.';
            return;
        }

        summaryText.textContent = item.summary;
        summaryTrend.textContent = `trend: ${item.trend || '--'}`;
        summaryRisk.textContent = `risk: ${item.risk_level || '--'}`;
        summaryConfidence.textContent = `confidence: ${fmt(item.confidence)}`;
        (item.main_drivers || []).slice(0, 5).forEach((driver) => {
            const row = document.createElement('li');
            row.textContent = `${driver.date || '--'} · ${driver.title || 'Новость'} · ${driver.why || ''}`;
            summaryDrivers.appendChild(row);
        });
    }

    function showLoadError() {
        clearTable(factorBody, 'Не удалось загрузить мониторинг.', 4, 'text-danger');
        clearTable(newsBody, 'Ошибка загрузки.', 3, 'text-danger');
        clearTable(spikeBody, 'Ошибка загрузки.', 4, 'text-danger');
    }

    function loadMonitoringData(factorKey = selectedFactorKey) {
        const params = new URLSearchParams({ days: String(monitoringWindowDays), engine: selectedEngine });
        if (factorKey) params.set('factor', factorKey);
        return fetch(`/api/monitoring/?${params.toString()}`)
            .then((response) => response.json())
            .then((data) => {
                renderFeed(data);
                return loadSummary(selectedFactorKey);
            })
            .catch((error) => {
                console.error(error);
                showLoadError();
                throw error;
            });
    }

    function loadSummary(factorKey = selectedFactorKey) {
        if (selectedEngine !== 'llm') {
            renderSummary({ llm_enabled: false, factors: [] });
            return Promise.resolve();
        }
        const params = new URLSearchParams({ engine: 'llm', period: 'month' });
        if (factorKey) params.set('factor', factorKey);
        return fetch(`/api/monitoring/summary/?${params.toString()}`)
            .then((response) => response.json())
            .then(renderSummary)
            .catch((error) => {
                console.error(error);
                summaryText.textContent = 'Не удалось загрузить LLM-summary.';
            });
    }

    function refreshButtonLabel() {
        refreshBtn.textContent = selectedEngine === 'llm' ? 'Запустить LLM-анализ' : 'Обновить новости сейчас';
    }

    function pollTask(taskId, options = {}) {
        return fetch(`/api/task/${taskId}/`)
            .then((response) => response.json())
            .then((status) => {
                if (status.status === 'Complete') {
                    setActionStatus(options.successMessage || 'Задача завершена, данные обновляются.', 'success');
                    return loadMonitoringData(selectedFactorKey);
                }
                if (status.status === 'Failed') {
                    throw new Error(status.error || 'Задача завершилась ошибкой');
                }
                const label = status.status === 'Running'
                    ? (options.runningLabel || 'Задача выполняется...')
                    : 'Ожидание запуска...';
                if (options.button) options.button.textContent = label;
                setActionStatus(label, 'running');
                return new Promise((resolve) => {
                    window.setTimeout(() => resolve(pollTask(taskId, options)), 1500);
                });
            });
    }

    factorSelect.addEventListener('change', (event) => {
        selectedFactorKey = event.target.value;
        loadMonitoringData(selectedFactorKey).catch(() => setActionStatus('Не удалось загрузить фактор.', 'error'));
    });

    engineSelect.addEventListener('change', (event) => {
        selectedEngine = event.target.value;
        refreshButtonLabel();
        loadMonitoringData(selectedFactorKey).catch(() => setActionStatus('Не удалось переключить источник анализа.', 'error'));
    });

    refreshBtn.addEventListener('click', () => {
        setActionButtonsDisabled(true);
        refreshBtn.textContent = 'Обновляем...';
        setActionStatus('Запущено обновление свежих новостей.', 'running');

        fetch('/api/monitoring/run/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken(),
            },
            body: JSON.stringify({
                engine: selectedEngine,
                force: false,
                summarize: selectedEngine === 'llm',
                period: 'day',
                async: true,
            }),
        })
            .then((response) => response.json())
            .then((payload) => {
                if (payload.task_id) {
                    return pollTask(payload.task_id, {
                        button: refreshBtn,
                        runningLabel: selectedEngine === 'llm' ? 'LLM анализирует...' : 'Новости обновляются...',
                        successMessage: 'Свежие новости обработаны.',
                    });
                }
                return loadMonitoringData(selectedFactorKey);
            })
            .catch((error) => {
                console.error(error);
                setActionStatus(error.message || 'Не удалось запустить мониторинг.', 'error');
                summaryText.textContent = error.message || 'Не удалось запустить мониторинг.';
            })
            .finally(() => {
                setActionButtonsDisabled(false);
                refreshButtonLabel();
            });
    });

    historyBackfillBtn?.addEventListener('click', () => {
        setActionButtonsDisabled(true);
        historyBackfillBtn.textContent = 'Заполняем историю...';
        setActionStatus('Запущен архивный backfill за 365 дней. Это может идти долго.', 'running');

        fetch('/api/monitoring/history/run/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken(),
            },
            body: JSON.stringify({
                engine: selectedEngine,
                days: 365,
                sources: ['interfax'],
                limit_per_day: 8,
                force: false,
                summarize: false,
                period: 'day',
                async: true,
            }),
        })
            .then((response) => response.json())
            .then((payload) => {
                if (payload.task_id) {
                    return pollTask(payload.task_id, {
                        button: historyBackfillBtn,
                        runningLabel: 'История заполняется...',
                        successMessage: 'Исторический backfill завершен.',
                    });
                }
                return loadMonitoringData(selectedFactorKey);
            })
            .catch((error) => {
                console.error(error);
                setActionStatus(error.message || 'Не удалось запустить исторический backfill.', 'error');
            })
            .finally(() => {
                setActionButtonsDisabled(false);
                historyBackfillBtn.textContent = 'Заполнить историю за год';
                refreshButtonLabel();
            });
    });

    reloadBtn.addEventListener('click', () => {
        setActionStatus('Экран перечитывает текущие агрегаты.', 'running');
        loadMonitoringData(selectedFactorKey)
            .then(() => setActionStatus('Экран обновлен.', 'success'))
            .catch(() => setActionStatus('Не удалось обновить экран.', 'error'));
    });
    window.addEventListener('themechange', () => {
        if (lastFeed) {
            renderOverviewChart(lastFeed.timeline || []);
            renderFactorChart(lastFeed.factor_detail?.timeline || [], lastFeed.factor_detail?.name);
        }
    });

    refreshButtonLabel();
    loadMonitoringData().catch(() => setActionStatus('Не удалось загрузить мониторинг.', 'error'));
});
