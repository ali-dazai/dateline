import { useEffect, useState } from 'react'
import { getAnswer, getPlain, type Answer, type Group } from './api'

// Dates are handled as their written parts, never converted through the browser's time zone.
const WD = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
const MO = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

export function parts(iso: string) {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  const wd = new Date(Date.UTC(y, m - 1, d)).getUTCDay()
  return { y, m, d, weekday: WD[wd], wd3: WD[wd].slice(0, 3), month: MO[m - 1], mon3: MO[m - 1].slice(0, 3) }
}

export const longDay = (iso: string) => { const p = parts(iso); return `${p.weekday}, ${p.month} ${p.d}, ${p.y}` }
export const shortDay = (iso: string) => { const p = parts(iso); return `${p.month} ${p.d}` }
export const ukDay = (iso: string) => { const p = parts(iso); return `${p.weekday} ${p.d} ${p.month} ${p.y}` }
export const clock = (iso: string) => iso.slice(11, 16)

/** Hours between two ISO moments (with offsets). */
export const hoursBetween = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 3.6e6

/** An ISO moment as local wall-clock time at a fixed UTC offset, e.g. '2023-06-21T14:00:00-04:00'. */
export function toLocal(iso: string, offset: number) {
  const t = new Date(Date.parse(iso) + offset * 3.6e6).toISOString().slice(0, 19)
  const h = Math.abs(offset), sign = offset < 0 ? '-' : '+'
  return `${t}${sign}${String(Math.floor(h)).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}`
}

export function addDays(iso: string, n: number) {
  const p = parts(iso)
  const t = new Date(Date.UTC(p.y, p.m - 1, p.d + n))
  return t.toISOString().slice(0, 10)
}

/** Split a context string around the quoted hit. */
export function split(context: string, hit: string) {
  const i = hit ? context.indexOf(hit) : -1
  if (i < 0) return { a: context, b: '', c: '' }
  return { a: context.slice(0, i), b: hit, c: context.slice(i + hit.length) }
}

export function cite(s: { paper: string; place: string; date: string; page: string }) {
  const bits = [s.paper]
  if (s.place && !s.place.startsWith('Post')) bits.push(s.place)
  if (s.date) bits.push(shortDay(s.date))
  if (s.page) bits.push(`page ${s.page}`)
  return bits.join(', ')
}

export const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`

/** "3 papers"; "about 3 papers" (or "1 paper, maybe more") when the count rests partly on a model's
 * reading of article openings. */
export const papers = (g: Group) =>
  !g.estimated ? plural(g.count, 'paper') : g.count === 1 ? '1 paper, maybe more' : `about ${plural(g.count, 'paper')}`

/** Claims shown under "Where the papers disagreed": the largest few (the engine sorts by count). */
export const GROUPS_SHOWN = 5

/** The plain model's reply as paragraphs, with Markdown marks (**, #, list bullets) taken off. */
export function paragraphs(text: string) {
  return text.split(/\n+/)
    .map(line => line.replace(/^\s*(#{1,6}\s+|[-*•]\s+)/, '').replace(/\*\*|__/g, '').replace(/\*(\S[^*]*?)\*/g, '$1').trim())
    .filter(Boolean)
}

export type PlainState = { loading: boolean; answer: string; moment: string; notice: string; error: string }

/** On demand: what the same model says when only told the moment. Forgets the reply when the question
 * or the moment changes, and ignores a reply that arrives after that. */
export function usePlain(story: string, question: string, cutoff: string) {
  const id = `${story}|${cutoff}|${question}`
  const [state, setState] = useState<PlainState & { id: string } | null>(null)
  const ask = () => {
    setState({ id, loading: true, answer: '', moment: '', notice: '', error: '' })
    const done = (patch: Partial<PlainState>) => setState(s => (s && s.id === id ? { ...s, loading: false, ...patch } : s))
    getPlain(story, question, cutoff)
      .then(r => done({ answer: r.answer, moment: r.moment, notice: r.notice }))
      .catch((e: Error) => done({ error: e.message }))
  }
  return { plain: state && state.id === id ? state : null, ask }
}

/** `asks` counts submits, so asking the same question again (after a failure) sends it again. While a
 * request runs, `data` is still the previous answer (pages dim it); a failure drops it, so an old answer
 * never sits under a new moment. */
export function useAnswer(story: string, question: string, cutoff: string, guard: boolean, asks = 0) {
  const [state, setState] = useState<{ data: Answer | null; error: string | null; loading: boolean }>(
    { data: null, error: null, loading: true })
  useEffect(() => {
    let live = true
    setState(s => ({ ...s, loading: true, error: null }))
    getAnswer(story, question, cutoff, guard)
      .then(data => { if (live) setState({ data, error: null, loading: false }) })
      .catch((e: Error) => { if (live) setState({ data: null, error: e.message, loading: false }) })
    return () => { live = false }
  }, [story, question, cutoff, guard, asks])
  return state
}

export function usePrefersReducedMotion() {
  const [r, setR] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => {
    const m = window.matchMedia('(prefers-reduced-motion: reduce)')
    const f = () => setR(m.matches)
    m.addEventListener('change', f)
    return () => m.removeEventListener('change', f)
  }, [])
  return r
}
