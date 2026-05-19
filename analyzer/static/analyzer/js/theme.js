(() => {
    const root = document.documentElement;
    const optionButtons = Array.from(document.querySelectorAll('[data-theme-value]'));
    const toggleButtons = Array.from(document.querySelectorAll('[data-theme-toggle]'));

    function normalizeTheme(value) {
        return value === 'light' ? 'light' : 'dark';
    }

    function updateControls(theme) {
        optionButtons.forEach((button) => {
            const isActive = button.dataset.themeValue === theme;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-pressed', String(isActive));
        });

        toggleButtons.forEach((button) => {
            button.textContent = theme === 'light' ? 'Белая тема' : 'Темная тема';
        });
    }

    function setTheme(theme) {
        const nextTheme = normalizeTheme(theme);
        root.dataset.theme = nextTheme;
        localStorage.setItem('theme', nextTheme);
        updateControls(nextTheme);
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: nextTheme } }));
    }

    const currentTheme = normalizeTheme(root.dataset.theme);
    updateControls(currentTheme);

    optionButtons.forEach((button) => {
        button.addEventListener('click', () => setTheme(button.dataset.themeValue));
    });

    toggleButtons.forEach((button) => {
        button.addEventListener('click', () => {
            setTheme(root.dataset.theme === 'light' ? 'dark' : 'light');
        });
    });
})();
