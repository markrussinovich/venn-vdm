export class VennClient {
  constructor(endpoint = "http://127.0.0.1:8000") {
    const url = new URL(endpoint);
    if (!["http:", "https:"].includes(url.protocol)) throw new Error("Use HTTP(S)");
    this.endpoint = endpoint.replace(/\/$/, "");
  }

  async decide(state, questions, { model = "venn-1", experimentalMedia = false } = {}) {
    const headers = { "content-type": "application/json" };
    if (experimentalMedia) headers["X-Venn-Experimental"] = "media";
    const response = await fetch(`${this.endpoint}/api/alpha/decisions`, {
      method: "POST", headers, body: JSON.stringify({ model, state, questions }),
    });
    if (!response.ok) throw new Error(`Venn HTTP ${response.status}: ${await response.text()}`);
    return response.json();
  }
}
