"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  DEMO_BOOKING, MONTHS, ROOM_TYPES, RESEARCH_WARNING,
  formatScore, parseBooking, parseModelResponse,
  type ModelResponse, type ResearchBooking,
} from "@/lib/research";

type Status = "checking" | "online" | "offline";
type Comparison = { with_year: ModelResponse; without_year: ModelResponse };

const metrics = [
  { label: "VALIDATION RECORDS", value: "14,517", helper: "Oct–Dec 2016 arrivals" },
  { label: "BEST RANKING / ROC AUC", value: "0.730", helper: "NumPy MLP · no year" },
  { label: "BEST PROBABILITY / BRIER", value: "0.203", helper: "NumPy MLP · with year" },
  { label: "AUTOMATED TESTS", value: "77", helper: "Passed locally" },
];

const comparisons = [
  { name: "Constant baseline", auc: 0.5, brier: 0.2368, tint: "muted" },
  { name: "Logistic · with year", auc: 0.7182, brier: 0.2175, tint: "secondary" },
  { name: "Logistic · without year", auc: 0.7172, brier: 0.2064, tint: "secondary" },
  { name: "NumPy MLP · with year", auc: 0.7207, brier: 0.2032, tint: "lime" },
  { name: "NumPy MLP · without year", auc: 0.73, brier: 0.2072, tint: "lime" },
];

const countInputs: { name: keyof ResearchBooking; label: string; min: number; max: number }[] = [
  { name: "lead_time", label: "Lead time (days)", min: 0, max: 10000 },
  { name: "stays_in_weekend_nights", label: "Weekend nights", min: 0, max: 1000 },
  { name: "stays_in_week_nights", label: "Weekday nights", min: 0, max: 1000 },
  { name: "adults", label: "Adults", min: 0, max: 1000 },
  { name: "children", label: "Children", min: 0, max: 1000 },
  { name: "babies", label: "Babies", min: 0, max: 1000 },
];

function SectionEyebrow({ children }: { children: React.ReactNode }) {
  return <span className="eyebrow"><span className="eyebrow-line" />{children}</span>;
}

function ScoreCard({ result, highlighted }: { result: ModelResponse; highlighted?: boolean }) {
  const score = result.research_cancellation_score;
  return (
    <article className={`score-card${highlighted ? " score-card-accent" : ""}`}>
      <div className="score-head">
        <span className="tiny-label">{result.variant === "without_year" ? "MODEL A · NO YEAR INPUT" : "MODEL B · ARRIVAL YEAR"}</span>
        <span className="mini-pill">NUMPY MLP</span>
      </div>
      <div className="score-content">
        <div className="score-dial" style={{ background: `conic-gradient(${highlighted ? "#d8f58f" : "#a3c2d4"} ${score * 100}%, #23343d 0)` }} aria-label={`Raw research score ${formatScore(score)}`}>
          <div className="score-dial-inner">
            <span>RAW SCORE</span>
            <strong>{formatScore(score)}</strong>
          </div>
        </div>
        <div className="score-copy">
          <strong>{result.variant === "without_year" ? "No arrival year" : "With arrival year"}</strong>
          <p>{result.variant === "without_year" ? "Year is excluded from model features." : "Year is included in model features."}</p>
          <span>Research only · not calibrated</span>
        </div>
      </div>
    </article>
  );
}

export function Dashboard() {
  const [booking, setBooking] = useState<ResearchBooking>({ ...DEMO_BOOKING });
  const [status, setStatus] = useState<Status>("checking");
  const [result, setResult] = useState<Comparison | null>(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [copied, setCopied] = useState(false);
  const requestNonce = useRef(0);

  async function refreshStatus() {
    try {
      const response = await fetch("/api/research/status", { cache: "no-store" });
      const body = (await response.json()) as { connected?: boolean };
      setStatus(response.ok && body.connected === true ? "online" : "offline");
    } catch {
      setStatus("offline");
    }
  }

  useEffect(() => { void refreshStatus(); }, []);

  function update<K extends keyof ResearchBooking>(key: K, value: ResearchBooking[K]) {
    requestNonce.current += 1; // Ignore an in-flight response for earlier inputs.
    setIsSubmitting(false);
    setBooking((previous) => ({ ...previous, [key]: value }));
    setResult(null);
    setError("");
    setCopied(false);
  }

  function reset() {
    requestNonce.current += 1;
    setIsSubmitting(false);
    setBooking({ ...DEMO_BOOKING });
    setResult(null);
    setError("");
    setCopied(false);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const parsed = parseBooking(booking);
    if (!parsed) {
      setError("Enter a valid historical synthetic booking (July 2015–December 2016). Check your dates and quantities.");
      setResult(null);
      return;
    }
    const nonce = ++requestNonce.current;
    setIsSubmitting(true);
    setError("");
    setResult(null);
    setCopied(false);
    try {
      const response = await fetch("/api/research/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ booking: parsed }),
      });
      const raw = await response.json();
      if (!response.ok) throw new Error(raw?.error || "The local inference service is unavailable.");
      const a = parseModelResponse(raw?.models?.without_year, "without_year");
      const b = parseModelResponse(raw?.models?.with_year, "with_year");
      if (!a || !b) throw new Error("The model response was incomplete.");
      if (nonce !== requestNonce.current) return;
      setResult({ without_year: a, with_year: b });
      setStatus("online");
    } catch (cause) {
      if (nonce !== requestNonce.current) return;
      setError(cause instanceof Error ? cause.message : "Could not retrieve model results.");
      await refreshStatus();
    } finally {
      if (nonce === requestNonce.current) setIsSubmitting(false);
    }
  }

  async function copyResults() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify({ input: booking, models: result, warning: RESEARCH_WARNING }, null, 2));
      setCopied(true);
    } catch {
      setError("Clipboard access was denied. You can still view the results on this page.");
    }
  }

  return (
    <div className="site-wrap" id="top">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />
      <header className="site-header container">
        <a href="#top" className="brand" aria-label="BookingForge home">
          <span className="brand-symbol" aria-hidden="true"><span/><span/><span/><span/></span>
          <span className="brand-name">booking<span>forge</span><small> / LAB</small></span>
        </a>
        <nav className="nav-links" aria-label="Main navigation">
          <a href="#sandbox">Sandbox</a>
          <a href="#evidence">Research</a>
          <a href="#architecture">Architecture</a>
        </nav>
        <button type="button" className={`connection connection-${status}`} onClick={() => void refreshStatus()} title="Check local FastAPI connection" aria-label={`Model service status: ${status}. Refresh.`}>
          <span className="status-dot" aria-hidden="true" />
          {status === "checking" ? "Checking service" : status === "online" ? "Models online" : "API offline"}
          <span className="connection-refresh" aria-hidden="true">↻</span>
        </button>
      </header>

      <main>
        <section className="hero container">
          <div className="hero-copy">
            <SectionEyebrow>APPLIED ML ENGINEERING / 001</SectionEyebrow>
            <h1>Hotel intelligence.<br/><span>Without the black box.</span></h1>
            <p>An interactive research environment for exploring neural networks trained from scratch on historical hotel reservations. Real architecture. Reproducible experiments. Transparent limits.</p>
            <div className="hero-actions">
              <a className="btn btn-accent" href="#sandbox">Explore the sandbox <span aria-hidden="true">↗</span></a>
              <a className="btn btn-ghost" href="#evidence">View research <span aria-hidden="true">↘</span></a>
            </div>
            <div className="hero-foot"><span className="hero-foot-dot" /> LOCAL RESEARCH ENVIRONMENT <span className="hero-foot-divider" /> NUMPY · FASTAPI · NEXT.JS</div>
          </div>
          <div className="hero-visual" aria-label="Illustration of the machine learning research system">
            <div className="hero-visual-top"><span>MODEL TELEMETRY</span><span className="mono">BF-MLP-01</span></div>
            <div className="signal-grid">
              <div className="signal-lines" aria-hidden="true">
                {[40, 80, 116, 152, 185, 207, 230, 248, 264, 284, 298, 310, 320, 329, 340, 346, 352, 358].map((h, index) => (
                  <span key={index} style={{ height: `${h / 4.3}%`, opacity: (index + 5) / 24 }} />
                ))}
              </div>
              <div className="hero-visual-focus">
                <span>RANKING SIGNAL</span>
                <strong>0.730<span> AUC</span></strong>
                <small>RESEARCH COHORT / OCT–DEC 2016</small>
              </div>
            </div>
            <div className="hero-visual-bottom"><span>01 / HISTORICAL DATA</span><span>02 / TRAINED MODEL</span><span>03 / INFERENCE</span></div>
          </div>
        </section>

        <section className="metrics-strip container" aria-label="Verified project metrics">
          {metrics.map((metric) => <div className="metric" key={metric.label}>
            <span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.helper}</small>
          </div>)}
        </section>

        <section id="sandbox" className="workspace-section container">
          <div className="section-heading">
            <div><SectionEyebrow>01 / THE SANDBOX</SectionEyebrow><h2>Run a synthetic scenario<span className="heading-period">.</span></h2><p>Change historical booking-like inputs and compare the two saved research models side by side.</p></div>
            <div className="section-aside">LOCAL INFERENCE<br/><span>NO RETRAINING</span></div>
          </div>
          <div className="workspace-grid">
            <section className="panel form-panel" aria-label="Synthetic booking input">
              <div className="panel-head"><div><span className="panel-index">/ A</span><h3>Booking parameters</h3></div><span className="panel-caption">SYNTHETIC INPUT ONLY</span></div>
              <form onSubmit={(event) => void submit(event)}>
                <div className="form-section-title">PROPERTY & ARRIVAL <span>01 / 03</span></div>
                <div className="form-grid">
                  <label className="field field-full">Hotel type
                    <select value={booking.hotel} onChange={e => update("hotel", e.target.value as ResearchBooking["hotel"])}>
                      <option>City Hotel</option><option>Resort Hotel</option>
                    </select>
                  </label>
                  <label className="field">Arrival year
                    <select value={booking.arrival_date_year} onChange={e => update("arrival_date_year", Number(e.target.value) as ResearchBooking["arrival_date_year"])}>
                      <option value={2015}>2015</option><option value={2016}>2016</option>
                    </select>
                  </label>
                  <label className="field">Arrival month
                    <select value={booking.arrival_date_month} onChange={e => update("arrival_date_month", e.target.value)}>
                      {MONTHS.map(month => <option key={month} value={month}>{month}</option>)}
                    </select>
                  </label>
                  <label className="field">Day of month
                    <input type="number" min={1} max={31} required value={booking.arrival_date_day_of_month} onChange={e => update("arrival_date_day_of_month", Number(e.target.value))} />
                  </label>
                  <label className="field">Reserved room type
                    <select value={booking.reserved_room_type} onChange={e => update("reserved_room_type", e.target.value)}>
                      {ROOM_TYPES.map(room => <option key={room} value={room}>Type {room}</option>)}
                    </select>
                  </label>
                </div>
                <div className="form-section-title">STAY DETAILS <span>02 / 03</span></div>
                <div className="form-grid">
                  {countInputs.slice(0, 3).map(field => <label className="field" key={field.name}>{field.label}
                    <input type="number" required min={field.min} max={field.max} step={1} value={booking[field.name]} onChange={e => update(field.name, Number(e.target.value))} />
                  </label>)}
                </div>
                <div className="form-section-title">GUEST COMPOSITION <span>03 / 03</span></div>
                <div className="form-grid">
                  {countInputs.slice(3).map(field => <label className="field" key={field.name}>{field.label}
                    <input type="number" required min={field.min} max={field.max} step={1} value={booking[field.name]} onChange={e => update(field.name, Number(e.target.value))} />
                  </label>)}
                </div>
                {error && <p className="form-error" role="alert">{error}</p>}
                <div className="form-actions">
                  <button type="submit" disabled={isSubmitting} className="btn btn-accent btn-submit">{isSubmitting ? "Running models…" : "Compare research models"}<span aria-hidden="true">↗</span></button>
                  <button type="button" className="btn-reset" onClick={reset}>Reset sample</button>
                </div>
              </form>
            </section>

            <section className="panel results-panel" aria-live="polite" aria-label="Research model scores">
              <div className="panel-head"><div><span className="panel-index">/ B</span><h3>Model outputs</h3></div><span className="panel-caption">RAW SCORES · 0–1</span></div>
              {result ? <>
                <div className="result-notice"><span className="result-notice-dot" />LIVE LOCAL INFERENCE <span>·</span> TWO SAVED MODELS</div>
                <div className="scores-stack"><ScoreCard result={result.without_year} highlighted /><ScoreCard result={result.with_year} /></div>
                <div className="difference-card"><span>ABSOLUTE SCORE DIFFERENCE</span><strong>{((Math.abs(result.without_year.research_cancellation_score - result.with_year.research_cancellation_score)) * 100).toFixed(2)} <small>percentage points</small></strong><p>Different feature sets may produce different raw scores. This comparison does not establish which output is correct.</p></div>
                <button type="button" className="copy-btn" onClick={() => void copyResults()}>{copied ? "✓ JSON copied" : "Copy synthetic input & results ↗"}</button>
              </> : <div className="empty-state">
                <div className="empty-visual" aria-hidden="true"><div className="empty-circle circle-one"/><div className="empty-circle circle-two"/><div className="empty-center">BF<span>·</span></div></div>
                <span className="tiny-label">AWAITING INPUT</span>
                <h4>Ready when you are.</h4>
                <p>Submit the sample booking to load both trained neural networks and display their raw research scores here.</p>
                <div className="empty-flow"><span>INPUT</span><i/> <span>FASTAPI</span><i/><span>NUMPY MLP</span></div>
              </div>}
              <div className="warning-panel"><span className="warning-symbol">!</span><p><strong>Research limitation</strong>{RESEARCH_WARNING}</p></div>
            </section>
          </div>
        </section>

        <section className="evidence-section container" id="evidence">
          <div className="section-heading"><div><SectionEyebrow>02 / THE EVIDENCE</SectionEyebrow><h2>Measured, not marketed<span className="heading-period">.</span></h2><p>Fixed retrospective benchmark on 14,517 historical arrivals. Higher ROC AUC indicates better ranking; lower Brier indicates smaller probability errors.</p></div><div className="section-aside">RESEARCH RESULTS<br/><span>NOT LIVE PERFORMANCE</span></div></div>
          <div className="research-grid">
            <div className="panel research-chart">
              <div className="panel-head"><div><span className="panel-index">/ R01</span><h3>Ranking quality</h3></div><span className="panel-caption">ROC AUC ↑</span></div>
              <div className="bar-chart-head"><span>MODEL</span><span>ROC AUC</span></div>
              <div className="bar-chart-list">{comparisons.map(item => <div className="bar-row" key={item.name}>
                <div className="bar-row-header"><span>{item.name}</span><strong>{item.auc.toFixed(4)}</strong></div>
                <div className="bar-track"><div className={`bar-fill bar-${item.tint}`} style={{ width: `${item.auc * 100}%` }}/></div>
              </div>)}</div>
              <p className="chart-caption">AUC of 0.50 represents constant/random ranking, not a useful discriminative model. These are retrospective research metrics.</p>
            </div>
            <div className="panel science-panel">
              <div className="panel-head"><div><span className="panel-index">/ R02</span><h3>Probability quality</h3></div><span className="panel-caption">BRIER ↓</span></div>
              <div className="science-highlight"><small>BEST OBSERVED BRIER</small><strong>0.2032</strong><span>MLP · WITH ARRIVAL YEAR</span></div>
              <div className="science-comparison"><div><span>Without year</span><strong>0.2072</strong></div><div><span>Constant baseline</span><strong>0.2368</strong></div></div>
              <div className="science-caution"><span>↳</span><p>Better ranking and better probability error do not identify the same model. Both scores are descriptive of this historical validation period.</p></div>
            </div>
          </div>
          <div className="research-footnotes"><span>59k HISTORICAL TRAINING RECORDS</span><span>2015–2016 ARRIVAL COHORTS</span><span>DATASET: HOTEL BOOKING DEMAND</span></div>
        </section>

        <section id="architecture" className="architecture-section container">
          <div className="section-heading"><div><SectionEyebrow>03 / HOW IT WORKS</SectionEyebrow><h2>Built end to end<span className="heading-period">.</span></h2><p>From hand-written gradient updates to a versioned HTTP inference contract.</p></div></div>
          <div className="architecture-grid">
            {[
              { no: "01", name: "RESEARCH DATA", tech: "Pandas · temporal cohorts", description: "119,390 original historical records; outcome-maturity and leakage audits; no real-time data feeds." },
              { no: "02", name: "ORIGINAL MODELS", tech: "NumPy · backprop · Adam", description: "Custom neural network and exact kNN; gradient checks, paired benchmarks and documented limitations." },
              { no: "03", name: "SAVED ARTIFACTS", tech: "NPZ · JSON · SHA256", description: "Pickle-free weights, verified preprocessing metadata, identical predictions after reload." },
              { no: "04", name: "LOCAL API + UI", tech: "FastAPI · Next.js", description: "Typed synthetic inputs, two locally loaded model variants, same-origin browser proxy and research-only outputs." },
            ].map((step) => <article className="arch-card" key={step.no}><span>{step.no} / MODULE</span><strong>{step.name}</strong><small>{step.tech}</small><p>{step.description}</p></article>)}
          </div>
          <div className="method-note"><span className="method-glyph">✳</span><div><strong>An honest limitation is part of the engineering.</strong><p>The source does not contain immutable booking-time feature snapshots, and earlier exploratory audits touched nominal holdout outcomes. This demo cannot claim prospective cancellation accuracy or support real-world hotel operations.</p></div></div>
        </section>
      </main>

      <footer className="site-footer container"><a href="#top" className="brand"><span className="brand-symbol" aria-hidden="true"><span/><span/><span/><span/></span><span className="brand-name">booking<span>forge</span></span></a><span>REPRODUCIBLE ML RESEARCH / LOCAL DEMO</span><a href="#top">BACK TO TOP ↑</a></footer>
    </div>
  );
}
