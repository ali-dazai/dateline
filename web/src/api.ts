// Calls to the Dateline API (api/server.py) and the shapes the pages use.
// API: GET /api/health, GET /api/events, GET /api/range?event=, POST /api/answer (contract v2, dateline/contract.py),
// POST /api/plain (the same model told the moment, no search).
import { STORY_META } from './stories'

export type Theme = 'plates' | 'highlighter' | 'abyss'

export interface Day { date: string; count: number | null }
export interface Stop { at: string; label: string }
/** A question asked at a moment (data/presets.json). `cutoff` is in the page's own format. */
export interface Preset { question: string; cutoff: string; label?: string }
export interface Hour { at: string; count: number }

export interface Story {
  id: string
  title: string
  year: number
  theme: Theme
  dates_label: string
  hook: string
  question: string
  unit: 'day' | 'minute'
  /** Range shown on the page. Minute stories use local-time ISO strings with an offset. */
  start: string
  end: string
  default_cutoff: string
  source: string
  days?: Day[]
  hours?: Hour[]
  stops?: Stop[]
  markers?: Stop[]
  presets?: Preset[]
  timezone_label?: string
  /** Hours from UTC for the minute story's clock (e.g. -4 for US Eastern summer time). */
  utc_offset?: number
  sample: boolean
}

export interface Sentence {
  text: string
  quote: string
  article_id: string
  paper: string
  place: string
  date: string
  published: string
  page: string
  url: string
  context: string
  hit: string
  match: number
  group: number
  unsourced?: boolean
  /** Quote check off: why the check would have removed this sentence. */
  flag?: string
}

/** `count` is the number of distinct papers; `estimated` when it rests partly on a model's reading of openings. */
export interface Group { claim: string; count: number; papers: string[]; article_ids: string[]; estimated: boolean }
export interface Chart {
  title: string
  unit: string
  min: number
  max: number
  points: { label: string; value: number; group: number }[]
}

export interface Answer {
  story: string
  cutoff: string
  guard: boolean
  sentences: Sentence[]
  removed: { text: string; reason: string }[]
  groups: Group[]
  retrieved: string[]
  chart: Chart | null
  /** A message the reader must see (nothing published yet, the model failed, every sentence removed...). */
  notice: string
  problems: string[]
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error('the API on port 8000 did not respond')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      // FastAPI: a string, a list of {msg} (422), or an object such as {message, error_class}.
      const d = (await res.json()).detail
      detail = typeof d === 'string' ? d
        : Array.isArray(d) ? d.map(x => x?.msg).filter(Boolean).join('; ')
        : typeof d?.message === 'string' ? d.message : JSON.stringify(d)
    } catch { /* not JSON */ }
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

interface ApiEvent { id: string; title: string; dates: string; start: string; end: string; granularity: 'day' | 'minute'; look: string; source: string }
interface ApiRange { event: string; min: string; max: string; granularity: string; counts: { bucket: string; total: number; story: number }[] }

const THEMES: Record<string, Theme> = { 'sunday-plates': 'plates', highlighter: 'highlighter', abyss: 'abyss' }

function eachDay(start: string, end: string) {
  const out: string[] = []
  for (let t = Date.parse(start.slice(0, 10) + 'T00:00:00Z'); t <= Date.parse(end.slice(0, 10) + 'T00:00:00Z'); t += 864e5) {
    out.push(new Date(t).toISOString().slice(0, 10))
  }
  return out
}

/** The stories the API serves, with the page details (question, hook, stops) from stories.ts. */
export async function getStories(): Promise<Story[]> {
  const [events, health] = await Promise.all([
    call<ApiEvent[]>('/api/events'),
    call<{ ok: boolean; engine: string }>('/api/health'),
  ])
  const ranges = await Promise.all(events.map(e => call<ApiRange>(`/api/range?event=${encodeURIComponent(e.id)}`).catch(() => null)))
  return events.map((e, i) => {
    const meta = STORY_META[e.id] ?? {}
    const counts = ranges[i]?.counts ?? []
    const theme = meta.theme ?? THEMES[e.look] ?? 'plates'
    const base: Story = {
      id: e.id, title: e.title, year: Number(e.start.slice(0, 4)), theme, dates_label: e.dates, hook: '',
      question: `What happened in ${e.title}?`, unit: e.granularity, start: e.start, end: e.end,
      default_cutoff: e.granularity === 'day' ? e.start : e.start, source: e.source, sample: health.engine === 'fake',
      ...meta,
    }
    if (base.unit === 'day') {
      const byDay = new Map(counts.map(c => [c.bucket.slice(0, 10), c.story]))
      base.days = eachDay(e.start, e.end).map(d => ({ date: d, count: byDay.has(d) ? byDay.get(d)! : null }))
    } else {
      base.hours = counts.map(c => ({ at: c.bucket.replace(/Z$/, ':00Z'), count: c.story }))
    }
    return base
  })
}

interface ApiSentence { text: string; quote: string; article_id: string; paper: string; date: string; page: string; url: string;
  published?: string; group?: number; place?: string; context?: string; hit?: string; match?: number; flag?: string }
interface ApiAnswer { sentences: ApiSentence[]; removed: { text: string; reason: string }[];
  groups: { claim: string; count: number; article_ids: string[]; papers?: string[]; estimated?: boolean }[]; retrieved: string[];
  chart?: Chart | null; notice?: string; problems?: string[] }

/** Ask the engine; fills the display fields an engine may leave out (context, hit, papers per group). */
export async function getAnswer(story: string, question: string, cutoff: string, guard: boolean): Promise<Answer> {
  const r = await call<ApiAnswer>('/api/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event: story, question, cutoff, guard }),
  })
  const groupOf = (s: ApiSentence) => {
    if (typeof s.group === 'number') return s.group
    const g = r.groups.findIndex(x => x.article_ids.includes(s.article_id))
    return g
  }
  const sentences: Sentence[] = r.sentences.map(s => {
    const unsourced = !s.article_id
    return {
      text: s.text, quote: unsourced ? '' : s.quote, article_id: s.article_id, paper: s.paper, place: s.place ?? '',
      date: s.date, published: s.published || s.date, page: s.page, url: s.url,
      context: s.context || s.quote, hit: s.hit || s.quote, match: s.match ?? 100,
      group: groupOf(s), unsourced, flag: s.flag || undefined,
    }
  })
  const groups: Group[] = r.groups.map(g => ({
    claim: g.claim, count: g.count, article_ids: g.article_ids, estimated: !!g.estimated,
    papers: g.papers ?? [...new Set(sentences.filter(s => g.article_ids.includes(s.article_id)).map(s => s.paper))],
  }))
  return { story, cutoff, guard, sentences, removed: r.removed, groups, retrieved: r.retrieved, chart: r.chart ?? null, notice: r.notice ?? '', problems: r.problems ?? [] }
}

/** `moment` is the wording the model was told, e.g. "the end of Monday, April 15, 1912". */
export interface Plain { answer: string; moment: string; notice: string }

/** What the same model says when only told the moment: no search, no sources. */
export async function getPlain(story: string, question: string, cutoff: string): Promise<Plain> {
  const r = await call<{ answer: string; moment?: string; notice?: string }>('/api/plain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event: story, question, cutoff }),
  })
  return { answer: r.answer ?? '', moment: r.moment ?? '', notice: r.notice ?? '' }
}
