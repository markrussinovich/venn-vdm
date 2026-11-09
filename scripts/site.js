/* Static documentation search; all URLs are relative to the generated index. */
(() => {
  const script = document.currentScript;
  const base = new URL(".", script.src);
  const search = document.querySelector(".docs-search");
  const input = document.querySelector("#docs-query");
  const results = document.querySelector("#search-results");
  const status = document.querySelector("#search-status");
  const entries = window.VENN_SEARCH;
  if (search && Array.isArray(entries)) {
    search.hidden = false;
    const indexed = entries.map(entry => ({
      ...entry,
      heading: `${entry.title} ${entry.section}`.toLowerCase(),
      body: entry.text.toLowerCase(),
    }));
    const renderResults = () => {
      results.replaceChildren();
      const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) {
        status.textContent = "";
        return;
      }
      // One search over all pages; on user pages, maintainer pages rank after equally good user matches.
      const audience = location.pathname.includes("/maintainers/") ? "maintainer" : "user";
      const matches = indexed.filter(entry => terms.every(term =>
        entry.heading.includes(term) || entry.body.includes(term)
      )).map(entry => ({
        entry,
        score: terms.reduce((score, term) => score + (entry.heading.includes(term) ? 10 : 1), 0)
          + (entry.audience === audience ? 0.5 : 0),
      })).sort((a, b) => b.score - a.score);
      status.textContent = matches.length
        ? `${matches.length} results${matches.length > 20 ? "; showing the first 20" : ""}.`
        : "No results. Try another word or browse the sections below.";
      for (const { entry } of matches.slice(0, 20)) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = new URL(entry.url, base).href;
        link.textContent = entry.section ? `${entry.title} › ${entry.section}` : entry.title;
        const snippet = document.createElement("p");
        const position = entry.body.indexOf(terms.find(term => entry.body.includes(term)) || "");
        const start = Math.max(0, position - 45);
        snippet.textContent = `${start ? "…" : ""}${entry.text.slice(start, start + 150)}${entry.text.length > start + 150 ? "…" : ""}`;
        item.append(link, snippet);
        results.append(item);
      }
    };
    input.addEventListener("input", renderResults);
    input.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        input.value = "";
        input.dispatchEvent(new Event("input"));
      } else if (event.key === "Enter") {
        results.querySelector("a")?.focus();
      }
    });
    document.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        input.focus();
      }
    });
  }
  const sidebar = document.querySelector(".docs-sidebar");
  if (sidebar) {
    // Remember where the sidebar was scrolled so the next page opens it at the same place.
    const remember = () => { try { sessionStorage.setItem("venn-sidebar", String(sidebar.scrollTop)); } catch (e) {} };
    sidebar.addEventListener("scroll", remember, { passive: true });
    window.addEventListener("pagehide", remember);
  }
  const menu = document.querySelector(".docs-menu");
  const mobile = window.matchMedia("(max-width: 980px)");
  if (menu) {
    const updateMenu = () => { menu.open = !mobile.matches; };
    updateMenu();
    mobile.addEventListener("change", updateMenu);
  }
  const contents = document.querySelector(".page-contents");
  if (contents) {
    const compact = window.matchMedia("(max-width: 1200px)");
    const updateContents = () => { contents.open = !compact.matches; };
    updateContents();
    compact.addEventListener("change", updateContents);
  }
})();
