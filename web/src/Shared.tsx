import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import type { Preset, Story } from './api'
import { paragraphs, type PlainState } from './lib'

export type Go = (target: string) => void

/** Enter in a question box asks the question (as it does in a one-line input); Shift+Enter adds a line. */
export function submitOnEnter(e: KeyboardEvent<HTMLTextAreaElement>) {
  if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return
  e.preventDefault()
  e.currentTarget.form?.requestSubmit()
}

export interface ThemeProps {
  story: Story
  stories: Story[]
  go: Go
}

/** Story links at the top of each story page. `t` prefixes the theme's class names. */
export function StoryNav({ t, story, stories, go }: ThemeProps & { t: string }) {
  return (
    <nav aria-label="Stories" className={`${t}-nav`}>
      <button type="button" className={`${t}-navlink ${t}-navlink--muted`} onClick={() => go('start')}>All stories</button>
      {stories.map(s => (
        <button
          key={s.id}
          type="button"
          className={`${t}-navlink`}
          aria-current={s.id === story.id ? 'page' : undefined}
          onClick={() => s.id !== story.id && go(s.id)}
        >
          {s.title.replace(/^The /, '')}, {s.year}
        </button>
      ))}
    </nav>
  )
}

export function QuoteToggle({ t, on, set }: { t: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <button type="button" className={`${t}-toggle`} aria-pressed={on} onClick={() => set(!on)}>
      <span className={`${t}-toggle-label`}>Quote check</span>
      <span aria-hidden="true" className={`${t}-toggle-track ${on ? 'is-on' : ''}`}><span className={`${t}-toggle-knob`} /></span>
      <span className={`${t}-toggle-word`}>{on ? 'On' : 'Off'}</span>
    </button>
  )
}

export function SampleNote({ t, story }: { t: string; story: Story }) {
  if (!story.sample) return null
  return (
    <p className={`${t}-sample`}>
      These answers come from the stand-in engine, built from real {story.year} reporting, until the full engine is switched on.
    </p>
  )
}

/** Shown while an answer is on its way; the previous answer stays below, dimmed. */
export function LoadingNote({ t, text }: { t: string; text: string }) {
  return <p className={`${t}-sample`} role="status">{text}</p>
}

export function ErrorNote({ t, error }: { t: string; error: string }) {
  return (
    <p className={`${t}-error`} role="alert">
      The engine did not answer: {error}.
    </p>
  )
}

/** Preset questions (data/presets.json): choosing one fills the question and moves the moment. */
export function Presets({ t, presets, isCurrent, pick, when, color }: {
  t: string; presets: Preset[]; isCurrent: (p: Preset) => boolean; pick: (p: Preset) => void
  when: (cutoff: string) => string; color?: (i: number) => string
}) {
  if (!presets.length) return null
  return (
    <div className={`${t}-presets`}>
      <span className={`${t}-presetlabel`} id={`${t}-presets-l`}>Questions to try</span>
      <div className={`${t}-presetrow`} role="group" aria-labelledby={`${t}-presets-l`}>
        {presets.map((p, i) => (
          <button
            key={`${p.cutoff}|${p.question}`}
            type="button"
            className={`${t}-preset`}
            aria-pressed={isCurrent(p)}
            onClick={() => pick(p)}
            style={color ? ({ '--pen': color(i) } as CSSProperties) : undefined}
          >
            <span className={`${t}-presetq`}>{p.label || p.question}</span>
            <span className={`${t}-presetwhen`}>{when(p.cutoff)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/** On demand: the same model told only the moment, shown beside Dateline's answer. `when` is the moment in
 * words, used until the API says what the model was told. A long reply is clamped to a few lines with a
 * toggle (`moreClass` styles it like the page's other "show more" buttons). */
export function PlainAI({ t, plain, ask, when, moreClass }: {
  t: string; plain: PlainState | null; ask: () => void; when: string; moreClass?: string
}) {
  const shown = plain && !plain.loading && plain.answer
  const note = plain && !plain.loading && !plain.answer ? (plain.notice || `The plain model did not answer: ${plain.error}.`) : ''
  const body = useRef<HTMLDivElement>(null)
  const [whole, setWhole] = useState(false)
  const [long, setLong] = useState(false)
  const reply = shown ? plain.answer : ''
  useEffect(() => setWhole(false), [reply])
  useLayoutEffect(() => {
    const el = body.current
    setLong(!!el && (whole || el.scrollHeight > el.clientHeight + 1))
  }, [reply, whole])
  return (
    <section className={`${t}-plain ${shown ? 'is-shown' : ''}`} aria-label="A plain AI, told the date" aria-live="polite"
      aria-busy={plain?.loading || undefined}>
      {shown ? (
        <>
          <div className={`${t}-plainhead`}>
            <h2>What a plain AI says</h2>
            <p className={`${t}-plainsub`}>The same model Dateline uses, told it is {plain.moment || when}. It searched nothing and cites no sources.</p>
          </div>
          <div className={`${t}-plaintext`}>
            <div ref={body} className={`${t}-plainbody ${whole ? '' : 'is-clamped'} ${long && !whole ? 'is-faded' : ''}`}>
              {paragraphs(plain.answer).map((line, i) => <p key={i}>{line}</p>)}
            </div>
          </div>
          {long && (
            <button type="button" className={moreClass ?? `${t}-more`} aria-expanded={whole} onClick={() => setWhole(!whole)}>
              {whole ? 'Show less' : 'Read the whole reply'}
            </button>
          )}
        </>
      ) : (
        <>
          <button type="button" className={`${t}-plainbtn`} onClick={ask} disabled={!!plain?.loading}>
            {plain?.loading ? 'Asking a plain AI…' : 'What a plain AI says'}
          </button>
          <p className={`${t}-plainsub`}>{note || 'The same model, told only the date, with nothing to read.'}</p>
        </>
      )}
    </section>
  )
}
