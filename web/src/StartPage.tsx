import type { Story } from './api'
import './start.css'

export const START_HEADER = 168
/** Band height: must match `.st-band` in start.css. */
export const bandHeight = () => Math.max(200, (window.innerHeight - START_HEADER) / 3)

function rnd(i: number) { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x) }
const SNOW = Array.from({ length: 16 }, (_, k) => {
  const i = k + 50, s = 1.5 + rnd(i) * 2.5
  return {
    left: `${(rnd(i + 99) * 100).toFixed(2)}%`, top: `${(rnd(i + 7) * 100).toFixed(2)}%`, width: s, height: s,
    opacity: 0.15 + rnd(i + 3) * 0.35, animationDuration: `${(16 + rnd(i + 5) * 22).toFixed(1)}s`, animationDelay: `-${(rnd(i + 11) * 20).toFixed(1)}s`,
  }
})

function PlatesArt({ year }: { year: number }) {
  const t = (fill: string, cls: string) => (
    <g className={`st-bpl ${cls}`}><text x="750" y="226" textAnchor="end" className="st-year" fill={fill}>{year}</text></g>
  )
  return (
    <svg aria-hidden="true" className="st-art" viewBox="0 0 760 244">
      <defs>
        <pattern id="st-pC" width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(15)"><circle cx="5.5" cy="5.5" r="3.6" fill="#0098D4" /></pattern>
        <pattern id="st-pM" width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(75)"><circle cx="5.5" cy="5.5" r="3.9" fill="#E0067A" /></pattern>
      </defs>
      {t('#FFDD00', 'st-bplY')}{t('url(#st-pM)', 'st-bplM')}{t('url(#st-pC)', 'st-bplC')}
      <text x="750" y="226" textAnchor="end" className="st-year" fill="none" stroke="#161616" strokeWidth="2">{year}</text>
    </svg>
  )
}

function HighlighterArt() {
  return (
    <span aria-hidden="true" className="st-art st-sheets">
      <span className="st-sheet st-sheetA">
        <span className="st-sheetpaper">Reuters, September 12</span>
        <span className="st-sheethead st-sheethead--big" style={{ background: '#FFE83A' }}>Lehman rescue hopes rise</span>
      </span>
      <span className="st-sheet st-sheetB">
        <span className="st-sheetpaper">The Wall Street Journal, September 14</span>
        <span className="st-sheethead" style={{ background: '#FF74BC' }}>Barclays Emerges as a Leader In Talks Over Lehman Brothers</span>
      </span>
    </span>
  )
}

function AbyssArt() {
  return (
    <span aria-hidden="true" className="st-art st-lock">
      <span className="st-lockclock">14:00</span>
      <span className="st-note">
        <span className="st-notewho">US Coast Guard Northeast</span>
        <span className="st-notetext">Canadian P-3 aircraft detected underwater noises in the search area.</span>
      </span>
    </span>
  )
}

export default function StartPage({ stories, go, intro, error }: { stories: Story[]; go: (id: string) => void; intro: boolean; error: string | null }) {
  return (
    <section className="st" aria-label="Pick a story">
      <div className="st-header">
        <div className="st-titles">
          <div className="st-wordmark">Dateline</div>
          <h1>Pick a story</h1>
        </div>
        <p>Then choose a moment. Dateline answers only from what was published by then.</p>
      </div>
      {error && <p className="st-error" role="alert">Could not load the stories: {error}. Start the API with <code>.venv\Scripts\python.exe -m uvicorn api.server:app --port 8000</code>, then reload.</p>}
      <div className={`st-bands ${intro ? 'is-intro' : ''}`}>
        {stories.map(s => (
          <button key={s.id} type="button" className={`st-band st-band--${s.theme}`} onClick={() => go(s.id)}>
            {s.theme === 'abyss' && <span className="st-snow" aria-hidden="true">{SNOW.map((f, i) => <span key={i} style={f} />)}</span>}
            <span className="st-text">
              <span className="st-title">{s.title}</span>
              <span className="st-dates">{s.dates_label}</span>
              <span className="st-hook">{s.hook}</span>
            </span>
            {s.theme === 'plates' && <PlatesArt year={s.year} />}
            {s.theme === 'highlighter' && <HighlighterArt />}
            {s.theme === 'abyss' && <AbyssArt />}
          </button>
        ))}
      </div>
    </section>
  )
}
