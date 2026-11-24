import { useState, type CSSProperties } from 'react'
import type { Preset } from '../api'
import { GROUPS_SHOWN, cite, longDay, papers, parts, split, useAnswer, usePlain } from '../lib'
import { ErrorNote, LoadingNote, PlainAI, Presets, QuoteToggle, SampleNote, StoryNav, submitOnEnter, type ThemeProps } from '../Shared'
import './highlighter.css'

const PENS = ['#FFE83A', '#FF74BC', '#7CF08A', '#FFA94D']
const pen = (g: number) => PENS[((g % 4) + 4) % 4]

function Paperclip({ color, style }: { color: string; style?: CSSProperties }) {
  return (
    <svg aria-hidden="true" width="28" height="70" viewBox="0 0 28 70" style={style}>
      <path d="M9 6 V52 A9 9 0 0 0 27 52 V14 A6 6 0 0 0 15 14 V48" fill="none" stroke={color} strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  )
}

const short = (t: string, n = 76) => (t.length > n ? t.slice(0, n).replace(/\s\S*$/, '') + '…' : t)

export default function Highlighter(props: ThemeProps) {
  const { story } = props
  const days = story.days ?? []
  const [cutoff, setCutoff] = useState(story.default_cutoff)
  const [guard, setGuard] = useState(true)
  const [draft, setDraft] = useState(story.question)
  const [asked, setAsked] = useState(story.question)
  const [focus, setFocus] = useState(-1)
  const [asks, setAsks] = useState(0)
  const { data, error, loading } = useAnswer(story.id, asked, cutoff, guard, asks)
  const { plain, ask } = usePlain(story.id, asked, cutoff)
  const [allFor, setAllFor] = useState('')
  const idx = Math.max(0, days.findIndex(d => d.date === cutoff))
  const counted = days.some(d => d.count)
  const maxN = Math.max(1, ...days.map(d => d.count ?? 0))
  const p = parts(cutoff)
  const key = `${cutoff}-${guard}-${asked}`
  const fresh = data && data.cutoff.slice(0, 10) === cutoff
  const empty = data && data.sentences.length === 0
  const dim = (g: number) => focus >= 0 && g !== focus

  const pick = (d: string) => { setCutoff(d); setFocus(-1) }
  const choose = (pr: Preset) => { setDraft(pr.question); setAsked(pr.question); pick(pr.cutoff.slice(0, 10)) }
  const groups = data ? (allFor === key ? data.groups : data.groups.slice(0, GROUPS_SHOWN)) : []
  const hidden = data ? data.groups.length - groups.length : 0

  // The collage: the first three different articles the current answer quotes.
  const seen = new Set<string>()
  const clips = (data?.sentences ?? []).filter(s => !s.unsourced && !seen.has(s.article_id) && seen.add(s.article_id)).slice(0, 3)

  return (
    <div className="hl">
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden="true">
        <defs>
          <filter id="hl-toner" x="0" y="0" width="100%" height="100%">
            <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves={1} seed={4} />
            <feColorMatrix values="0 0 0 0 0.1  0 0 0 0 0.1  0 0 0 0 0.12  0 0 0 9 -6.2" />
          </filter>
        </defs>
      </svg>
      <div className="hl-wrap">
        <header className="hl-header">
          <div className="hl-brand">
            <div className="hl-wordmark">Dateline</div>
            <StoryNav t="hl" {...props} />
          </div>
          <QuoteToggle t="hl" on={guard} set={v => { setGuard(v); setFocus(-1) }} />
        </header>

        <section className="hl-hero" aria-label="Question">
          <form className="hl-form" onSubmit={e => { e.preventDefault(); if (draft.trim().length > 2) { setAsked(draft.trim()); setAsks(n => n + 1); setFocus(-1) } }}>
            <label htmlFor="hl-q" className="hl-label">Your question</label>
            <div className="hl-qwrap">
              <textarea id="hl-q" rows={2} value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={submitOnEnter} className="hl-q" />
              <svg aria-hidden="true" viewBox="0 0 600 24" preserveAspectRatio="none" className="hl-stroke">
                <path key={asked} className="hl-draw" d="M6 14 C120 9 260 16 400 11 S560 13 594 10" fill="none" stroke="#FFE83A" strokeWidth="14" strokeLinecap="round" opacity=".95" />
              </svg>
            </div>
            <div className="hl-formrow">
              <button type="submit" className="hl-ask">{loading ? 'Reading…' : 'Ask'}</button>
              <p className="hl-soft">Answers come only from reporting published on or before the day you choose.</p>
            </div>
            <Presets t="hl" presets={story.presets ?? []} pick={choose} color={pen}
              isCurrent={pr => pr.question === asked && pr.cutoff.slice(0, 10) === cutoff}
              when={c => { const d = parts(c); return `${d.weekday}, ${d.month} ${d.d}` }} />
          </form>
          <div className="hl-collage" aria-hidden="true" key={fresh ? key : 'wait'}>
            {(clips.length ? clips : [null]).map((s, i) => {
              return (
                <div key={i} className={`hl-pos hl-pos--${i}`}>
                  <div className="hl-deal" style={{ animationDelay: `${i * 140}ms` }}>
                    <div className={`hl-sheet hl-rot--${i} ${s ? '' : 'is-blank'}`}>
                      <svg width="100%" height="100%" className="hl-toner"><rect width="100%" height="100%" filter="url(#hl-toner)" /></svg>
                      <div className="hl-sheetpaper">{s ? `${s.paper}, ${parts(s.date).month} ${parts(s.date).d}` : 'Not published yet'}</div>
                      <div className="hl-sheethead">
                        {s && <span className="hl-swipe" style={{ backgroundImage: `linear-gradient(${pen(s.group)}, ${pen(s.group)})`, animationDelay: `${520 + i * 160}ms` }}>{short(s.hit || s.quote)}</span>}
                      </div>
                      <div className="hl-sheetlines" />
                    </div>
                  </div>
                </div>
              )
            })}
            <Paperclip color="#D9DBE6" style={{ position: 'absolute', left: 34, top: -22, transform: 'rotate(8deg)' }} />
          </div>
        </section>

        <section className="hl-days" aria-labelledby="hl-days-h">
          <div className="hl-sechead">
            <h2 id="hl-days-h">Choose a day</h2>
            <p className="hl-soft">Each stack is one day of reporting. The tabs on your day are the claims in the answer. Later days are still blank.</p>
          </div>
          <div className="hl-scroll">
            <div className="hl-rack">
              <div className="hl-cols">
                {days.map((d, i) => {
                  const printed = i <= idx
                  const dp = parts(d.date)
                  const h = !printed ? 0 : counted ? (d.count ? Math.round(16 + 196 * d.count / maxN) : 12) : 40 + i * 14
                  const delay = printed ? 0 : (i - idx - 1) * 70
                  const tabs = i === idx && fresh && data ? groups : []
                  return (
                    <button
                      key={d.date}
                      type="button"
                      className={`hl-col ${printed ? 'is-printed' : 'is-blank'} ${i === idx ? 'is-current' : ''}`}
                      aria-pressed={i === idx}
                      aria-label={printed ? `${longDay(d.date)}${d.count ? `, ${d.count} articles` : ''}` : `${dp.month} ${dp.d}, not published yet`}
                      onClick={() => pick(d.date)}
                    >
                      <span className="hl-face" aria-hidden="true" style={{ transitionDelay: `${delay}ms` }}>
                        <span className="hl-tabs">
                          {tabs.map((_, j) => <span key={j} className="hl-tab" style={{ background: pen(j), animationDelay: `${200 + j * 90}ms` }} />)}
                        </span>
                        <span className="hl-ink" style={{ transitionDelay: `${delay}ms` }}>
                          <span className="hl-i1" /><span className="hl-i2" /><span className="hl-i3" /><span className="hl-i4" />
                        </span>
                      </span>
                      <span className="hl-bundle" aria-hidden="true" style={{ height: h, transitionDelay: `${delay}ms` }} />
                      <span className="hl-collabel">
                        <span className="hl-wd">{dp.wd3}</span>
                        <span className="hl-date">{dp.mon3} {dp.d}</span>
                        <span className="hl-wd">{printed ? (d.count ? `${d.count.toLocaleString('en-US')} articles` : ' ') : 'Not yet'}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
              <div className="hl-rule" aria-hidden="true" style={{ left: `${((idx + 1) * 100) / Math.max(1, days.length)}%` }}>
                <span>Reports up to {p.month} {p.d}</span>
              </div>
            </div>
          </div>
        </section>

        {loading && <LoadingNote t="hl" text={`Reading the reports published up to ${p.month} ${p.d}…`} />}
        {error && <ErrorNote t="hl" error={error} />}
        <SampleNote t="hl" story={story} />

        {data && (
          <main className="hl-main" key={key} aria-busy={loading || undefined} style={loading ? { opacity: 0.35 } : undefined}>
            <section className="hl-answer hl-fade" aria-labelledby="hl-ans-h">
              <svg aria-hidden="true" width="100%" height="100%" className="hl-toner hl-toner--page"><rect width="100%" height="100%" filter="url(#hl-toner)" /></svg>
              <Paperclip color="#8C8FA3" style={{ position: 'absolute', right: 64, top: -24 }} />
              <h2 id="hl-ans-h" className="hl-anstitle">
                {empty ? (data.notice || `Nothing published by ${p.month} ${p.d} answers this yet`) : `By ${longDay(cutoff)}, the reports said`}
              </h2>
              {empty && !data.notice && <p className="hl-big-soft">No article dated {p.month} {p.d} or earlier answers this. Choose a later day.</p>}
              {!empty && data.notice && <p className="hl-big-soft">{data.notice}</p>}
              <ol className="hl-list">
                {data.sentences.map((s, i) => {
                  const q = split(s.context, s.hit)
                  return (
                    <li key={i}>
                      <p className="hl-sentence">{s.text}</p>
                      {s.unsourced ? (
                        <p className="hl-nosrc">
                          Nothing published by {p.month} {p.d} says this. With the quote check on, this sentence is removed.
                          {s.flag && <span className="hl-flag">{s.flag}</span>}
                        </p>
                      ) : (
                        <figure className="hl-clip">
                          <blockquote>
                            {q.a}
                            <span className="hl-swipe hl-swipe--inline" style={{ backgroundImage: `linear-gradient(${pen(s.group)}, ${pen(s.group)})`, opacity: dim(s.group) ? 0.3 : 1, animationDelay: `${450 + i * 120}ms` }}>{q.b}</span>
                            {q.c}
                          </blockquote>
                          <figcaption>
                            <span>{cite(s)}</span>
                            {s.url && <a href={s.url} target="_blank" rel="noopener noreferrer">Open the article</a>}
                            {s.match < 100 && <span>Matched {s.match}% against the source text</span>}
                          </figcaption>
                        </figure>
                      )}
                      {s.flag && !s.unsourced && (
                        <p className="hl-nosrc">
                          This quote does not back the sentence. With the quote check on, it is removed.
                          <span className="hl-flag">{s.flag}</span>
                        </p>
                      )}
                    </li>
                  )
                })}
              </ol>
              {data.removed.length > 0 && (
                <div className="hl-removed">
                  <h3>Removed by the quote check</h3>
                  {data.removed.map((r, i) => (
                    <div key={i}><p className="hl-struck">{r.text}</p><p className="hl-soft-ink">{r.reason}</p></div>
                  ))}
                </div>
              )}
            </section>

            <aside className="hl-split hl-fade">
              <PlainAI t="hl" plain={plain} ask={ask} when={`the end of ${longDay(cutoff)}`} />
              {data.groups.length > 0 && (
                <section className="hl-disagree" aria-labelledby="hl-split-h">
                  <div className="hl-splithead">
                    <h2 id="hl-split-h">Where the reports disagreed</h2>
                    <p className="hl-soft">Choose a claim to see only its quotes.</p>
                  </div>
                  <ul className="hl-groups">
                    {groups.map((g, i) => (
                      <li key={i}>
                        <button
                          type="button"
                          aria-pressed={focus === i}
                          onClick={() => setFocus(focus === i ? -1 : i)}
                          className={`hl-group ${focus === i ? 'is-focus' : ''} ${dim(i) ? 'is-dim' : ''}`}
                          style={{ borderColor: focus === i ? pen(i) : 'transparent' }}
                        >
                          <span className="hl-grouphead">
                            <span className="hl-groupclaim">
                              <svg aria-hidden="true" width="34" height="16" viewBox="0 0 34 16"><path d="M3 9 C10 6 22 11 31 7" fill="none" stroke={pen(i)} strokeWidth="10" strokeLinecap="round" /></svg>
                              <span>{g.claim}</span>
                            </span>
                            <span className="hl-soft">{papers(g)}</span>
                          </span>
                          <span className="hl-grouppapers">{g.papers.join(', ')}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                  {(hidden > 0 || allFor === key) && data.groups.length > GROUPS_SHOWN && (
                    <button type="button" className="hl-more" onClick={() => setAllFor(allFor === key ? '' : key)}>
                      {hidden > 0 ? `Show ${hidden} more ${hidden === 1 ? 'claim' : 'claims'}` : 'Show fewer claims'}
                    </button>
                  )}
                  {data.chart && (
                    <figure className="hl-chart">
                      <figcaption>{data.chart.title}</figcaption>
                      {data.chart.points.map((pt, i) => (
                        <div className="hl-chartrow" key={i} style={{ opacity: dim(pt.group) ? 0.35 : 1 }}>
                          <span className="hl-chartlabel">{pt.label}</span>
                          <span className="hl-charttrack">
                            <span className="hl-dot" style={{ left: `${((pt.value - data.chart!.min) / (data.chart!.max - data.chart!.min)) * 100}%`, background: pen(pt.group) }} />
                          </span>
                          <span className="hl-chartval">{pt.value.toLocaleString('en-US')}</span>
                        </div>
                      ))}
                    </figure>
                  )}
                </section>
              )}
            </aside>
          </main>
        )}

        <footer className="hl-footer">
          <p className="hl-soft">Every quote links to the article it came from.</p>
          <svg aria-hidden="true" width="150" height="20" viewBox="0 0 150 20">
            {PENS.map((c, i) => <path key={c} d={`M${4 + i * 38} 10 C${14 + i * 38} 8 ${26 + i * 38} 12 ${34 + i * 38} 9`} stroke={c} strokeWidth="10" strokeLinecap="round" fill="none" />)}
          </svg>
        </footer>
      </div>
    </div>
  )
}
