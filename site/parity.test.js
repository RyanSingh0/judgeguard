/* Assert the browser port reproduces Python's probabilities.
 *
 * A silent numerical divergence here would mean the live demo shows a different
 * model from the one the paper and the README describe -- which is exactly the
 * class of bug this project exists to complain about.
 *
 *     node site/parity.test.js
 */
import { readFileSync } from "node:fs";
import { BrowserGuard } from "./guard.js";

const model = JSON.parse(readFileSync(new URL("./model.json", import.meta.url), "utf8"));
const guard = new BrowserGuard(model);

let worst = 0, worstCase = null, fails = 0;
for (const c of model.parity_cases) {
  const got = guard.probability(c.context, c.question, c.answer);
  const err = Math.abs(got - c.p);
  if (err > worst) { worst = err; worstCase = c.label; }
  if (err > 1e-6) {
    fails++;
    console.log(`  FAIL ${c.label.padEnd(13)} python=${c.p.toFixed(8)} js=${got.toFixed(8)} Δ=${err.toExponential(2)}`);
  }
}

const n = model.parity_cases.length;
console.log(`\n  ${n - fails}/${n} parity cases match within 1e-6`);
console.log(`  worst deviation: ${worst.toExponential(3)} (${worstCase})`);

// Decisions must agree too -- a probability that is close but lands the other
// side of the threshold is still a different product.
let dfail = 0;
for (const c of model.parity_cases) {
  const py = c.p >= model.threshold ? "allow" : "block";
  const js = guard.guard(c.context, c.question, c.answer).decision;
  if (py !== js) { dfail++; console.log(`  DECISION MISMATCH ${c.label}: python=${py} js=${js}`); }
}
console.log(`  ${n - dfail}/${n} decisions agree`);

process.exit(fails || dfail ? 1 : 0);
