import { existsSync, readFileSync } from "node:fs";

const html = readFileSync("index.html", "utf8");
const visibleText = html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
const failures = [];

const requiredText = [
  "DriveMA",
  "Driving Vision-Language-Action Models with Verifiable Meta-Actions",
  "8.079",
  "Data efficiency",
  "Qualitative comparisons"
];

const requiredSnippets = [
  'id="overview"',
  'id="method"',
  'id="results"',
  'id="cases"',
  'id="citation"',
  "https://arxiv.org/pdf/2605.31271",
  "Data · Coming Soon",
  "Code · Coming Soon",
  "./static/results/side-intersection-h264.mp4",
  "./static/results/curvature-control-h264.mp4",
  "./static/results/cone-corridor-h264.mp4",
  "./static/results/maneuver-error-h264.mp4",
  "./static/results/sft-vs-rl-longitudinal-h264.mp4",
  "./static/results/sft-vs-rl-lateral-h264.mp4",
  "./static/results/cone-corridor-2-h264.mp4",
  "./static/results/maneuver-error-2-h264.mp4",
  "./static/images/drivema/arc.jpg",
  "./static/images/drivema/case1.jpg",
  "./static/images/drivema/RL_arc.jpg",
  "./static/images/drivema/efficiency-left.jpg",
  "./static/images/drivema/efficiency-right.jpg"
];

for (const text of requiredText) {
  if (!visibleText.includes(text)) failures.push(`Missing visible text: ${text}`);
}

for (const snippet of requiredSnippets) {
  if (!html.includes(snippet)) failures.push(`Missing required markup: ${snippet}`);
}

const hrefs = [...html.matchAll(/\b(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
for (const href of hrefs) {
  if (/^(https?:|mailto:|#)/.test(href)) continue;
  const path = href.replace(/^\.\//, "").split("#")[0].split("?")[0];
  if (path && !existsSync(path)) failures.push(`Referenced asset does not exist: ${href}`);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("DriveMA site verification passed.");
