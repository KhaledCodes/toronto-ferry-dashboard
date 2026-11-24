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

const recorded = [
  {date: '2026-09-05', day: 'Saturday', predicted: '100', actual: '125', error_pct: '20'},
  {date: '2026-09-30', day: 'Wednesday', predicted: '3586', actual: '', error_pct: ''},
];
const observed = [
  {date: '2026-09-04', redemptions: 10},
  {date: '2026-09-05', redemptions: 125},
  {date: '2026-09-06', redemptions: 200},
  {date: '2026-09-28', redemptions: 3243},
  {date: '2026-09-29', redemptions: 2717},
];
const rows = context.buildAccuracyRows(recorded, observed);
assert.equal(rows.map(r => r.date).join(','), '2026-09-30,2026-09-29,2026-09-28,2026-09-06,2026-09-05');
assert.equal(rows[0].actual, '');
assert.equal(rows[1].partial, true);
assert.equal(rows[1].actual, 2717);
assert.equal(rows[2].predicted, '');
assert.equal(rows[2].actual, 3243);
assert.equal(rows[2].error_pct, undefined);
assert.equal(rows[4].error_pct, '20');
assert.equal(recorded[1].actual, ''); // Source history remains intact.
assert.equal(context.hasNumber(''), false);
assert.equal(context.hasNumber(null), false);
assert.equal(context.hasNumber('0'), true);
const recovered = context.buildAccuracyRows([
  {date: '2026-09-28', predicted: '3000', actual: '', error_pct: ''},
  {date: '2026-09-29', predicted: '3000', actual: '', error_pct: ''},
], observed);
assert.equal(recovered[0].error_pct, ''); // Never grade a partial day.
assert.ok(recovered[1].error_pct > 7 && recovered[1].error_pct < 8);
console.log('Accuracy tracker checks passed: missing forecasts, recovered actuals, and partial days.');

const scored = (date, extra = {}) => ({
  date, predicted: '100', actual: '125', error_pct: '20', ...extra,
});
const windowRows = [
  scored('2026-11-25'), // Future forecasts never count.
  scored('2026-11-24'), // Today is not a completed calendar day.
  scored('2026-11-23', {partial: true}),
  scored('2026-11-22'),
  scored('2026-11-21', {actual: '', error_pct: ''}),
  scored('2026-11-20', {predicted: ''}),
  scored('2026-11-19', {error_pct: ''}),
  scored('2026-11-18', {error_pct: 'NaN'}),
  scored('2026-10-25'), // Inclusive start of the 30-day calendar window.
  scored('2026-10-24'),
];
const recent = context.recentScoredForecasts(windowRows, new Date('2026-11-24T17:00:00Z'));
assert.equal(recent.map(r => r.date).join(','), '2026-11-22,2026-10-25');
// UTC has rolled over; the Toronto window must stay unchanged.
assert.equal(context.recentScoredForecasts(windowRows, new Date('2026-11-25T02:00:00Z'))
  .map(r => r.date).join(','), '2026-11-22,2026-10-25');
assert.equal(context.recentScoredForecasts([scored('2026-10-24')], new Date('2026-11-24T17:00:00Z')).length, 0);
// The spring DST transition must not shorten the calendar window.
assert.equal(context.recentScoredForecasts([
  scored('2026-03-08'), scored('2026-03-07'), scored('2026-02-06'), scored('2026-02-05'),
], new Date('2026-03-08T07:30:00Z')).map(r => r.date).join(','), '2026-03-07,2026-02-06');
assert.equal(context.recentScoredForecasts([
  scored('2027-01-01'), scored('2026-12-31'), scored('2026-12-02'), scored('2026-12-01'),
], new Date('2027-01-01T17:00:00Z')).map(r => r.date).join(','), '2026-12-31,2026-12-02');
console.log('30-day accuracy checks passed: calendar boundaries, missing results, DST, and year rollover.');
