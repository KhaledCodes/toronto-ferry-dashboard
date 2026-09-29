const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const pending = () => new Promise(() => {});
const context = vm.createContext({
  d3: {format: () => {}, timeFormat: () => {}, csv: pending, json: pending,
       timeMonth: {}, timeYear: {}},
  window: {addEventListener: () => {}},
});
vm.runInContext(script, context);
const check = (date, now, extra = {}) => context.isCurrentForecast({
  prediction_date: date, predicted_redemptions: 100, ...extra,
}, new Date(now));
assert.equal(check('2026-09-05', '2026-09-29T21:00:00Z'), false);
assert.equal(check('2026-09-30', '2026-09-29T21:00:00Z'), true);
// UTC is already tomorrow; Toronto is still September 29.
assert.equal(check('2026-09-30', '2026-09-30T02:00:00Z'), true);
assert.equal(check('2026-10-01', '2026-09-30T02:00:00Z'), false);
assert.equal(check('2026-03-09', '2026-03-08T06:30:00Z'), true);
assert.equal(check('2026-11-02', '2026-11-01T05:30:00Z'), true);
assert.equal(check('2027-01-01', '2026-12-31T18:00:00Z'), true);
assert.equal(check('2026-09-30', '2026-09-29T21:00:00Z', {status: 'unavailable'}), false);
assert.equal(context.isCurrentForecast(null), false);
console.log('Dashboard freshness checks passed (including DST and Toronto midnight).');
