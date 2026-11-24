import { useState } from 'react'
import type { Preset } from '../api'
import { GROUPS_SHOWN, addDays, clock, hoursBetween, papers, parts, split, toLocal, ukDay, useAnswer, usePlain } from '../lib'
import { ErrorNote, LoadingNote, PlainAI, Presets, QuoteToggle, SampleNote, StoryNav, type ThemeProps } from '../Shared'
import './abyss.css'

function rnd(i: number) { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x) }
const SNOW = Array.from({ length: 42 }, (_, i) => {
  const s = 1.5 + rnd(i) * 2.5
  return {
    left: `${(rnd(i + 99) * 100).toFixed(2)}%`, top: `${(rnd(i + 7) * 100).toFixed(2)}%`, width: s, height: s,
    opacity: 0.15 + rnd(i + 3) * 0.35, animationDuration: `${(16 + rnd(i + 5) * 22).toFixed(1)}s`, animationDelay: `-${(rnd(i + 11) * 20).toFixed(1)}s`,
  }
})

function mono(name: string) {
  const words = name.replace(/\(.*?\)/g, '').split(/\s+/).filter(w => w && w !== 'The')
  if (!words.length) return '?'
  if (/^[A-Z]{2,4}$/.test(words[0])) return words[0]
  return words.slice(0, 2).map(w => w[0].toUpperCase()).join('')
}

function when(published: string, offset: number) {
  if (published.length <= 10) { const p = parts(published); return `${p.wd3} ${p.d} ${p.month}` }
  const local = toLocal(published, offset)
  return `${parts(local).wd3} ${clock(local)}`
}

const hourWord = (iso: string) => {
  const h = Number(iso.slice(11, 13)), m = iso.slice(14, 16)
  const h12 = h % 12 === 0 ? 12 : h % 12
  return `${h12}${m === '00' ? '' : ':' + m} ${h < 12 ? 'a.m.' : 'p.m.'}`
}

export default function Abyss(props: ThemeProps) {
  const { story } = props
  const stops = story.stops ?? []
  const markers = story.markers ?? []
  const [cutoff, setCutoff] = useState(story.default_cutoff)
  const [guard, setGuard] = useState(true)
  const [draft, setDraft] = useState(story.question)
  const [asked, setAsked] = useState(story.question)
  const [open, setOpen] = useState(false)
  const [asks, setAsks] = useState(0)
  const { data, error, loading } = useAnswer(story.id, asked, cutoff, guard, asks)
  const { plain, ask } = usePlain(story.id, asked, cutoff)
  const [allFor, setAllFor] = useState('')

  const off = story.utc_offset ?? 0
  const span = hoursBetween(story.start, story.end)
  const frac = (iso: string) => hoursBetween(story.start, iso) / span
  const y = (iso: string) => `${(frac(iso) * 100).toFixed(2)}%`
  const dayCount = Math.ceil(span / 24)
  const sel = stops.findIndex(s => s.at === cutoff)
  const isPast = (at: string) => Date.parse(at) < Date.parse(cutoff)
  const key = `${cutoff}-${guard}-${asked}`
  const p = parts(cutoff)
  const shortWhen = `${clock(cutoff)} on ${p.weekday} ${p.d} ${p.month}`
  const maxHour = Math.max(1, ...(story.hours ?? []).map(h => h.count))

  // Pick any moment on the rail, to the nearest quarter hour.
  const pickAt = (clientY: number, rail: HTMLElement) => {
    const r = rail.getBoundingClientRect()
    const f = Math.min(1, Math.max(0, (clientY - r.top) / r.height))
    const q = Math.round((f * span * 60) / 15) * 15
    const t = Math.min(Date.parse(story.end), Date.parse(story.start) + q * 6e4)
    setCutoff(toLocal(new Date(t).toISOString(), off))
    setOpen(false)
  }

  // Number each source in the order the answer first quotes it.
  const used: string[] = []
  for (const s of data?.sentences ?? []) if (!s.unsourced && !used.includes(s.article_id)) used.push(s.article_id)
  const sources = used.map(id => {
    const first = data!.sentences.find(s => s.article_id === id)!
    return { id, s: first, q: split(first.context, first.hit) }
  })
  const empty = data && data.sentences.length === 0
  const groups = data ? (allFor === key ? data.groups : data.groups.slice(0, GROUPS_SHOWN)) : []
  const hidden = data ? data.groups.length - groups.length : 0
  const choose = (pr: Preset) => { setDraft(pr.question); setAsked(pr.question); setCutoff(toLocal(pr.cutoff, off)); setOpen(false) }

  return (
    <div className="ab">
      <div className="ab-snow" aria-hidden="true">
        {SNOW.map((f, i) => <span key={i} style={f} />)}
      </div>
      <div className="ab-wrap">
        <header className="ab-header">
          <div className="ab-brand">
            <div className="ab-wordmark">Dateline</div>
            <StoryNav t="ab" {...props} />
          </div>
          <QuoteToggle t="ab" on={guard} set={setGuard} />
        </header>

        <div className="ab-body">
          <section className="ab-time" aria-labelledby="ab-time-h">
            <h2 id="ab-time-h">Choose a moment</h2>
            <p className="ab-soft ab-small">Time runs downward. Pick a preset or anywhere on the line. Everything below your moment has not been posted yet; the faint bars are how much was posted each hour.</p>
            <div className="ab-rail">
              <span className="ab-railhit" aria-hidden="true" onClick={e => pickAt(e.clientY, e.currentTarget.parentElement!)} />
              {(story.hours ?? []).map(h => (
                <span key={h.at} className="ab-ping" aria-hidden="true" style={{ top: y(toLocal(h.at, off)), width: 6 + 46 * h.count / maxHour }} />
              ))}
              {Array.from({ length: Math.floor(span / 6) + 1 }, (_, i) => (
                <span key={i} className={`ab-tick ${i % 4 === 0 ? 'is-day' : ''}`} style={{ top: `${((i * 6) / span) * 100}%` }} />
              ))}
              {Array.from({ length: dayCount }, (_, i) => {
                const dp = parts(addDays(story.start, i))
                return <span key={i} className="ab-daylabel" style={{ top: `${((i * 24) / span) * 100}%` }}>{dp.wd3} {dp.d}</span>
              })}
              <span className="ab-depth" style={{ height: y(cutoff) }} />
              <span className="ab-below" style={{ top: y(cutoff) }} />
              {markers.map(m => (
                <span key={m.at}>
                  <span className="ab-marker" style={{ top: y(m.at) }} aria-hidden="true" />
                  <span className="ab-markerlabel" style={{ top: y(m.at) }}>{m.label}</span>
                </span>
              ))}
              {stops.map((s, i) => (
                <button
                  key={s.at}
                  type="button"
                  className={`ab-stop ${i === sel ? 'is-current' : isPast(s.at) ? 'is-past' : ''}`}
                  style={{ top: y(s.at) }}
                  aria-pressed={i === sel}
                  aria-label={`${s.label}, ${ukDay(s.at)}`}
                  onClick={() => { setCutoff(s.at); setOpen(false) }}
                >
                  <span className="ab-stopdot"><span /></span>
                  <span className="ab-stoplabel">{s.label}</span>
                </button>
              ))}
              {sel < 0 && <span className="ab-thumb" style={{ top: y(cutoff) }} aria-hidden="true"><span /><b>{p.wd3} {clock(cutoff)}</b></span>}
              <span className="ab-notyet">Not posted yet</span>
            </div>
          </section>

          <main className="ab-main">
            <div className="ab-clock" key={cutoff}>
              <p className="ab-clockday">{ukDay(cutoff)}</p>
              <p className="ab-clocktime" aria-label={hourWord(cutoff)}>{clock(cutoff)}</p>
              <p className="ab-soft">{story.timezone_label ?? 'Local time'}. Dateline reads only what was published by this minute.</p>
            </div>

            <form className="ab-form" onSubmit={e => { e.preventDefault(); if (draft.trim().length > 2) { setAsked(draft.trim()); setAsks(n => n + 1) } }}>
              <label htmlFor="ab-q" className="ab-label">Your question</label>
              <div className="ab-field">
                <input id="ab-q" type="text" value={draft} onChange={e => setDraft(e.target.value)} />
                <button type="submit">{loading ? 'Reading…' : 'Ask'}</button>
              </div>
              <Presets t="ab" presets={story.presets ?? []} pick={choose}
                isCurrent={pr => pr.question === asked && toLocal(pr.cutoff, off) === cutoff}
                when={c => { const l = toLocal(c, off); return `${parts(l).wd3} ${clock(l)}` }} />
            </form>

            {loading && <LoadingNote t="ab" text={`Reading what was posted by ${shortWhen}…`} />}
            {error && <ErrorNote t="ab" error={error} />}
            <SampleNote t="ab" story={story} />

            {data && (
              <div className="ab-results" key={key} aria-busy={loading || undefined} style={loading ? { opacity: 0.35 } : undefined}>
                <section className="ab-answercol" aria-label="Answer">
                  <article className="ab-card ab-rise">
                    <div className="ab-cardhead">
                      <span aria-hidden="true" className="ab-avatar">D</span>
                      <span className="ab-cardwho">
                        <span className="ab-cardname">Dateline</span>
                        <span className="ab-cardsub">Answer as of {shortWhen}</span>
                      </span>
                    </div>
                    {!empty && data.notice && <p className="ab-empty">{data.notice}</p>}
                    {empty ? (
                      <p className="ab-empty">{data.notice || `Nothing posted by ${shortWhen} answers this yet. Choose a later moment.`}</p>
                    ) : (
                      <ol className="ab-sentences">
                        {data.sentences.map((s, i) => (
                          <li key={i}>
                            {s.text}{' '}
                            {s.unsourced
                              ? <span className="ab-tag ab-tag--none">no source</span>
                              : <span className="ab-tag" aria-label={`source ${used.indexOf(s.article_id) + 1}`}>{used.indexOf(s.article_id) + 1}</span>}
                            {s.flag && <span className="ab-flag">Held back when the quote check is on: {s.flag}</span>}
                          </li>
                        ))}
                      </ol>
                    )}
                  </article>
                  {data.removed.map((r, i) => (
                    <article key={i} className="ab-held ab-rise" style={{ animationDelay: '100ms' }}>
                      <span className="ab-heldtitle">Held back by the quote check</span>
                      <span className="ab-struck">{r.text}</span>
                      <span className="ab-soft ab-small">{r.reason}</span>
                    </article>
                  ))}
                  {data.groups.length > 0 && (
                    <section className="ab-groups" aria-labelledby="ab-split-h">
                      <h2 id="ab-split-h">Where the reports disagreed</h2>
                      {groups.map((g, i) => (
                        <div key={i} className="ab-group">
                          <span className="ab-grouphead"><span>{g.claim}</span><span className="ab-soft ab-small">{papers(g)}</span></span>
                          <span className="ab-soft ab-small">{g.papers.join(', ')}</span>
                        </div>
                      ))}
                      {(hidden > 0 || allFor === key) && data.groups.length > GROUPS_SHOWN && (
                        <button type="button" className="ab-stackbtn ab-more" onClick={() => setAllFor(allFor === key ? '' : key)}>
                          {hidden > 0 ? `Show ${hidden} more ${hidden === 1 ? 'claim' : 'claims'}` : 'Show fewer claims'}
                        </button>
                      )}
                    </section>
                  )}
                </section>

                <div className="ab-side">
                  <PlainAI t="ab" moreClass="ab-stackbtn ab-more" plain={plain} ask={ask} when={`${hourWord(cutoff)} ${story.timezone_label ?? 'local time'} on ${ukDay(cutoff)}`} />
                  {sources.length > 0 && (
                    <section className="ab-sources" aria-labelledby="ab-src-h">
                      <div className="ab-sourceshead">
                        <h2 id="ab-src-h">Sources</h2>
                        {sources.length > 1 && (
                          <button type="button" className="ab-stackbtn" aria-expanded={open} onClick={() => setOpen(!open)}>
                            {open ? 'Stack sources' : `Show all ${sources.length}`}
                          </button>
                        )}
                      </div>
                      <div className={`ab-stack ${open ? 'is-open' : ''}`}>
                        {sources.map(({ id, s, q }, i) => (
                          <article
                            key={id}
                            className="ab-source"
                            style={open
                              ? { marginTop: i ? 12 : 0, zIndex: 10 - i }
                              : { marginTop: i ? -86 : 0, transform: `scale(${1 - i * 0.045})`, opacity: i > 2 ? 0 : 1 - i * 0.2, zIndex: 10 - i }}
                            aria-hidden={!open && i > 0 ? true : undefined}
                          >
                            <div className="ab-sourcehead">
                              <span aria-hidden="true" className="ab-mono">{mono(s.paper)}</span>
                              <span className="ab-outlet">{i + 1}. {s.paper}</span>
                              <span className="ab-when">{when(s.published || s.date, off)}</span>
                            </div>
                            <p>{q.a}<mark>{q.b}</mark>{q.c}</p>
                            {s.url && (open || i === 0) && <a href={s.url} target="_blank" rel="noopener noreferrer" className="ab-link">Open the article</a>}
                          </article>
                        ))}
                      </div>
                    </section>
                  )}
                </div>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}
