import { useState } from 'react'
import type { Preset } from '../api'
import { GROUPS_SHOWN, cite, longDay, papers, parts, split, useAnswer, usePlain } from '../lib'
import { ErrorNote, LoadingNote, PlainAI, Presets, QuoteToggle, SampleNote, StoryNav, submitOnEnter, type ThemeProps } from '../Shared'
import './plates.css'

const INKS = ['pl-dotC', 'pl-dotM', 'pl-dotK']

function Registration({ size = 26, color = '#161616' }: { size?: number; color?: string }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 26 26">
      <circle cx="13" cy="13" r="7" fill="none" stroke={color} strokeWidth="1.2" />
      <path d="M13 0 V26 M0 13 H26" stroke={color} strokeWidth="1.2" />
    </svg>
  )
}

export default function Plates(props: ThemeProps) {
  const { story } = props
  const days = story.days ?? []
  const [cutoff, setCutoff] = useState(story.default_cutoff)
  const [guard, setGuard] = useState(true)
  const [draft, setDraft] = useState(story.question)
  const [asked, setAsked] = useState(story.question)
  const [asks, setAsks] = useState(0)
  const { data, error, loading } = useAnswer(story.id, asked, cutoff, guard, asks)
  const { plain, ask } = usePlain(story.id, asked, cutoff)
  const [allFor, setAllFor] = useState('')
  const idx = Math.max(0, days.findIndex(d => d.date === cutoff))
  const maxN = Math.max(1, ...days.map(d => d.count ?? 0))
  const p = parts(cutoff)
  const key = `${cutoff}-${guard}-${asked}`
  const empty = data && data.sentences.length === 0
  const maxG = data ? Math.max(1, ...data.groups.map(g => g.count)) : 1
  const groups = data ? (allFor === key ? data.groups : data.groups.slice(0, GROUPS_SHOWN)) : []
  const hidden = data ? data.groups.length - groups.length : 0
  const pick = (pr: Preset) => { setDraft(pr.question); setAsked(pr.question); setCutoff(pr.cutoff.slice(0, 10)) }

  return (
    <div className="pl">
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden="true">
        <defs>
          <pattern id="pl-pC" width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(15)"><circle cx="5.5" cy="5.5" r="3.6" fill="#0098D4" /></pattern>
          <pattern id="pl-pM" width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(75)"><circle cx="5.5" cy="5.5" r="3.9" fill="#E0067A" /></pattern>
        </defs>
      </svg>
      <div className="pl-wrap">
        <header className="pl-header">
          <div className="pl-brand">
            <div className="pl-wordmark">Dateline</div>
            <span className="pl-colorbar" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /></span>
          </div>
          <StoryNav t="pl" {...props} />
          <QuoteToggle t="pl" on={guard} set={setGuard} />
        </header>

        <section className="pl-hero" aria-label="Question">
          <span className="pl-reg pl-reg--l"><Registration /></span>
          <span className="pl-reg pl-reg--r"><Registration /></span>
          <form className="pl-form" onSubmit={e => { e.preventDefault(); if (draft.trim().length > 2) { setAsked(draft.trim()); setAsks(n => n + 1) } }}>
            <label htmlFor="pl-q" className="pl-label">Ask the papers</label>
            <textarea id="pl-q" rows={2} value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={submitOnEnter} className="pl-q" />
            <div className="pl-formrow">
              <button type="submit" className="pl-ask">{loading ? 'Reading…' : 'Ask'}</button>
              <p className="pl-soft">Answers come only from newspapers printed on or before the day you choose.</p>
            </div>
            <Presets t="pl" presets={story.presets ?? []} pick={pick}
              isCurrent={pr => pr.question === asked && pr.cutoff.slice(0, 10) === cutoff}
              when={c => { const d = parts(c); return `${d.weekday}, ${d.month} ${d.d}` }} />
          </form>
          <div className="pl-numeral" aria-hidden="true">
            <svg viewBox="0 0 520 330" key={cutoff}>
              <g className="pl-plate pl-plY"><text x="510" y="300" textAnchor="end" className="pl-numtext" fill="#FFDD00">{p.d}</text></g>
              <g className="pl-plate pl-plM"><text x="510" y="300" textAnchor="end" className="pl-numtext" fill="url(#pl-pM)">{p.d}</text></g>
              <g className="pl-plate pl-plC"><text x="510" y="300" textAnchor="end" className="pl-numtext" fill="url(#pl-pC)">{p.d}</text></g>
              <text x="510" y="300" textAnchor="end" className="pl-numtext" fill="none" stroke="#161616" strokeWidth="2.5">{p.d}</text>
            </svg>
            <div className="pl-numcap">{p.weekday}, {p.month} {p.y}</div>
          </div>
        </section>

        <section className="pl-days" aria-labelledby="pl-days-h">
          <div className="pl-sechead">
            <h2 id="pl-days-h">Choose a day</h2>
            <p className="pl-soft">Each bundle is one day of papers that mention the Titanic. Days after the one you choose are still blank plates.</p>
          </div>
          <div className="pl-scroll">
            <div className="pl-rack">
              <div className="pl-cols">
                {days.map((d, i) => {
                  const printed = i <= idx
                  const dp = parts(d.date)
                  const h = printed ? (d.count ? Math.round(16 + 196 * d.count / maxN) : 12) : 0
                  const delay = printed ? 0 : (i - idx - 1) * 70
                  const plate = (base: number) => ({ transitionDelay: `${printed ? base : delay + base}ms` })
                  return (
                    <button
                      key={d.date}
                      type="button"
                      className={`pl-col ${printed ? 'is-printed' : 'is-blank'} ${i === idx ? 'is-current' : ''}`}
                      aria-pressed={i === idx}
                      aria-label={printed ? `${longDay(d.date)}${d.count ? `, ${d.count} articles` : ''}` : `${dp.month} ${dp.d}, not printed yet`}
                      onClick={() => setCutoff(d.date)}
                    >
                      <span className="pl-face" aria-hidden="true">
                        <span className="pl-ink pl-inkK" style={plate(0)}>
                          <span className="pl-k1" /><span className="pl-k2" /><span className="pl-k3" /><span className="pl-k4" />
                        </span>
                        <span className="pl-ink pl-plate pl-inkM" style={plate(90)} />
                        <span className="pl-ink pl-plate pl-dotC pl-inkC" style={plate(180)} />
                        <span className="pl-ink pl-plate pl-inkY" style={plate(270)} />
                        {!printed && <span className="pl-blankmark"><Registration size={30} color="#8C877B" /></span>}
                      </span>
                      <span className="pl-bundle" aria-hidden="true" style={{ height: h, transitionDelay: `${delay}ms` }}><i /></span>
                      <span className="pl-collabel">
                        <span className="pl-wd">{dp.wd3}</span>
                        <span className="pl-date">{dp.mon3} {dp.d}</span>
                        <span className="pl-wd">{printed ? (d.count ? `${d.count.toLocaleString('en-US')} articles` : ' ') : 'Not printed yet'}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
              <div className="pl-rule" aria-hidden="true" style={{ left: `${((idx + 1) * 100) / days.length}%` }}>
                <span>Papers up to {parts(cutoff).month} {parts(cutoff).d}</span>
              </div>
            </div>
          </div>
        </section>

        {loading && <LoadingNote t="pl" text={`Reading the papers printed up to ${p.month} ${p.d}…`} />}
        {error && <ErrorNote t="pl" error={error} />}
        <SampleNote t="pl" story={story} />

        {data && (
          <main className="pl-main" key={key} aria-busy={loading || undefined} style={loading ? { opacity: 0.35 } : undefined}>
            <section className="pl-answer pl-fade" aria-labelledby="pl-ans-h">
              <h2 id="pl-ans-h" className="pl-anstitle">
                {empty ? (data.notice || `Nothing printed by ${p.month} ${p.d} answers this yet`) : `By ${longDay(cutoff)}, the papers said`}
              </h2>
              {empty && !data.notice && <p className="pl-big-soft">No article dated {p.month} {p.d} or earlier answers this. Choose a later day.</p>}
              {!empty && data.notice && <p className="pl-big-soft">{data.notice}</p>}
              <ol className="pl-list">
                {data.sentences.map((s, i) => {
                  const q = split(s.context, s.hit)
                  return (
                    <li key={i}>
                      <p className="pl-sentence">{s.text}</p>
                      {s.unsourced ? (
                        <p className="pl-nosrc">
                          No paper printed by {p.month} {p.d} says this. With the quote check on, this sentence is removed.
                          {s.flag && <span className="pl-flag">{s.flag}</span>}
                        </p>
                      ) : (
                        <figure className="pl-clip">
                          <blockquote>
                            {q.a}<span className="pl-swipe">{q.b}</span>{q.c}
                            <span className="pl-clipreg"><Registration size={18} /></span>
                          </blockquote>
                          <figcaption>
                            <span>{cite(s)}</span>
                            {s.url && <a href={s.url} target="_blank" rel="noopener noreferrer">Open the scan</a>}
                            {s.match < 100 && <span>Matched {s.match}% against the OCR text</span>}
                          </figcaption>
                        </figure>
                      )}
                      {s.flag && !s.unsourced && (
                        <p className="pl-nosrc">
                          This quote does not back the sentence. With the quote check on, it is removed.
                          <span className="pl-flag">{s.flag}</span>
                        </p>
                      )}
                    </li>
                  )
                })}
              </ol>
              {data.removed.length > 0 && (
                <div className="pl-removed">
                  <h3>Removed by the quote check</h3>
                  {data.removed.map((r, i) => (
                    <div key={i}><p className="pl-struck">{r.text}</p><p className="pl-soft">{r.reason}</p></div>
                  ))}
                </div>
              )}
            </section>

            <aside className="pl-split pl-fade">
              <PlainAI t="pl" plain={plain} ask={ask} when={`the end of ${longDay(cutoff)}`} />
              {data.groups.length > 0 && (
                <section className="pl-disagree" aria-labelledby="pl-split-h">
                  <h2 id="pl-split-h">Where the papers disagreed</h2>
                  {data.chart && (
                    <figure className="pl-chart">
                      <figcaption>{data.chart.title}</figcaption>
                      {data.chart.points.map((pt, i) => {
                        const x = ((pt.value - data.chart!.min) / (data.chart!.max - data.chart!.min)) * 100
                        return (
                          <div className="pl-chartrow" key={i}>
                            <span className="pl-chartlabel">{pt.label}</span>
                            <span className="pl-charttrack">
                              <span className="pl-plate pl-dot pl-dot--c" style={{ left: `${x}%` }} />
                              <span className="pl-plate pl-dot pl-dot--m" style={{ left: `${x}%` }} />
                            </span>
                            <span className="pl-chartval">{pt.value.toLocaleString('en-US')}</span>
                          </div>
                        )
                      })}
                      <div className="pl-chartaxis" aria-hidden="true">
                        <span />
                        <span className="pl-ticks">
                          {[0.125, 0.375, 0.625, 0.875].map(f => Math.round(data.chart!.min + f * (data.chart!.max - data.chart!.min))).map(v => (
                            <span key={v} style={{ left: `${((v - data.chart!.min) / (data.chart!.max - data.chart!.min)) * 100}%` }}>{v.toLocaleString('en-US')}</span>
                          ))}
                        </span>
                        <span />
                      </div>
                    </figure>
                  )}
                  <ul className="pl-groups">
                    {groups.map((g, i) => (
                      <li key={i}>
                        <span className="pl-grouphead"><span>{g.claim}</span><span className="pl-soft">{papers(g)}</span></span>
                        <span className="pl-bar" aria-hidden="true"><span className={INKS[i % 3]} style={{ width: `${(g.count / maxG) * 100}%` }} /></span>
                        <span className="pl-soft pl-small">{g.papers.join(', ')}</span>
                      </li>
                    ))}
                  </ul>
                  {(hidden > 0 || allFor === key) && data.groups.length > GROUPS_SHOWN && (
                    <button type="button" className="pl-more" onClick={() => setAllFor(allFor === key ? '' : key)}>
                      {hidden > 0 ? `Show ${hidden} more ${hidden === 1 ? 'claim' : 'claims'}` : 'Show fewer claims'}
                    </button>
                  )}
                </section>
              )}
            </aside>
          </main>
        )}

        <footer className="pl-footer">
          <p className="pl-soft">Every quote links to its page scan at the Library of Congress.</p>
          <span className="pl-colorbar pl-colorbar--small" aria-hidden="true"><i /><i /><i /><i /></span>
        </footer>
      </div>
    </div>
  )
}
