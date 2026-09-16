# Making these videos without breaching the policy they explain

The uncomfortable bit: this series is generative AI output about generative AI rules. It has to
be the most compliant thing Nteractive ships this year, because the first person to find a
shortcut in it gets to ignore every episode.

| # | Requirement | Clause | Applied here |
|---|---|---|---|
| 1 | Only approved AI tools | §5.1, §6.1 | Every image/video/voice generator used goes through Head of IT + Head of ESG review and onto the Trelica registry **before** the first render. A model that isn't on the list doesn't touch this project. |
| 2 | No confidential data into AI tools | §5.3 | Prompts contain policy text, which is internal. Treat the ISMS policy as Internal: draft prompts in an approved environment only, never in a free-tier or personal account. No client names, no staff names, no real screens — every insert shot is deliberately illegible. |
| 3 | Meaningful human review | §5.5 | Head of IT signs off the policy accuracy of every script before render, and the final cut before publish. The accuracy table in `ep01-work-vs-web-30s.md` is the review artefact. |
| 4 | No Shadow AI | §5.6 | No browser-extension editors, no "AI enhance" toggles inside whatever edit tool is used, unless IT has reviewed that specific feature. New AI feature in an existing tool = new approval. |
| 5 | Client-facing check | §5.7 | Internal-only distribution (Viva Engage, Teams, ISMS landing page). If a client ever asks for it, that's a new decision and it goes back through the account lead and IT. |
| 6 | Escalation | §5.9 | Any automation built to batch-render or auto-publish episodes is a system, not personal productivity — IT reviews it before it's built. |
| 7 | Consent and likeness | §6.2, general | If a real colleague appears instead of the synthetic host, written consent for the likeness and its retention. If the host is synthetic, the likeness resembles no identifiable person and carries a persistent `AI-generated presenter` mark. |
| 8 | Copyright | §4 Generative AI row | Music, fonts and stock licensed for commercial internal use. Generated assets logged with the tool, date and prompt so provenance can be answered later. |
| 9 | Exceptions | §10 | If any of the above can't be met, it's an exception: written approval from the Head of IT, and the risk on the risk register. Not a verbal nod in a stand-up. |

## Asset log — keep one row per generated asset

| Asset | Tool + version | Date | Prompt ref | Reviewed by | Approved |
|---|---|---|---|---|---|
| | | | | | |
