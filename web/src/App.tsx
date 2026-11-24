import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ComponentType, type CSSProperties, type ReactNode } from 'react'
import { getStories, type Story, type Theme } from './api'
import { usePrefersReducedMotion } from './lib'
import type { ThemeProps } from './Shared'
import StartPage, { START_HEADER, bandHeight } from './StartPage'
import Plates from './themes/Plates'
import Highlighter from './themes/Highlighter'
import Abyss from './themes/Abyss'
import './app.css'

const PAGES: Record<Theme, ComponentType<ThemeProps>> = { plates: Plates, highlighter: Highlighter, abyss: Abyss }

const hashView = () => decodeURIComponent(window.location.hash.slice(1)) || 'start'

interface Nav { view: string; from: string | null; n: number }

export default function App() {
  const [stories, setStories] = useState<Story[]>([])
  const [error, setError] = useState<string | null>(null)
  const [nav, setNav] = useState<Nav>({ view: hashView(), from: null, n: 0 })
  const [visited, setVisited] = useState<string[]>([])
  const [band, setBand] = useState({ bt: 0, bb: 0 })
  const [intro, setIntro] = useState(true)
  const reduced = usePrefersReducedMotion()
  const layers = useRef<Record<string, HTMLDivElement | null>>({})

  useEffect(() => {
    getStories().then(setStories).catch((e: Error) => setError(e.message))
  }, [])

  const byId = useCallback((id: string) => stories.find(s => s.id === id), [stories])
  const themeOf = useCallback((id: string | null) => (id === 'start' || id === null ? 'start' : byId(id)?.theme ?? 'start'), [byId])

  // Unknown story in the address bar: fall back to the start page.
  useEffect(() => {
    if ((stories.length || error) && nav.view !== 'start' && !byId(nav.view)) setNav(v => ({ ...v, view: 'start', from: null }))
  }, [stories, error, nav.view, byId])

  useEffect(() => {
    if (nav.view !== 'start' && !visited.includes(nav.view)) setVisited(v => [...v, nav.view])
    const s = byId(nav.view)
    document.title = s ? `Dateline: ${s.title}, ${s.year}` : 'Dateline'
  }, [nav.view, visited, byId])

  const navRef = useRef(nav)
  navRef.current = nav
  const navigate = useCallback((target: string) => {
    const cur = navRef.current
    if (target === cur.view) return
    // Where this story's band sits on the start page, as clip-path insets.
    const storyId = target === 'start' ? cur.view : cur.view === 'start' ? target : null
    if (storyId) {
      const i = Math.max(0, stories.findIndex(s => s.id === storyId))
      const el = cur.view === 'start' ? document.querySelectorAll<HTMLElement>('.st-band')[i] : undefined
      const r = el?.getBoundingClientRect()
      const top = r ? r.top : START_HEADER + i * bandHeight()
      const h = r ? r.height : bandHeight()
      setBand({ bt: Math.max(0, top), bb: Math.max(0, window.innerHeight - top - h) })
    }
    setNav({ view: target, from: reduced ? null : cur.view, n: cur.n + 1 })
    setIntro(false)
  }, [stories, reduced])

  const go = useCallback((target: string) => {
    if (target === nav.view) return
    window.history.pushState(null, '', target === 'start' ? window.location.pathname : `#${target}`)
    navigate(target)
  }, [nav.view, navigate])

  useEffect(() => {
    const onPop = () => navigate(hashView())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [navigate])

  // The arriving page starts at its top, so band positions and entrances line up.
  useLayoutEffect(() => {
    const el = layers.current[nav.view]
    if (el && nav.n > 0) el.scrollTop = 0
  }, [nav.view, nav.n])

  // Drop the leaving page once its animation is over (with a fallback if no animationend fires).
  const finish = useCallback((n: number) => setNav(cur => (cur.n === n ? { ...cur, from: null } : cur)), [])
  useEffect(() => {
    if (!nav.from) return
    const t = window.setTimeout(() => finish(nav.n), 1800)
    return () => window.clearTimeout(t)
  }, [nav.from, nav.n, finish])

  const vt = themeOf(nav.view)
  const ft = themeOf(nav.from)
  const cls = (id: string) => {
    if (id === nav.view) return nav.from ? `ly in-${vt}-from-${ft}` : 'ly ly-show'
    if (id === nav.from) return `ly ly-exit out-${ft}-to-${vt}`
    return 'ly ly-hidden'
  }
  const layer = (id: string, child: ReactNode) => (
    <div
      key={id}
      ref={el => { layers.current[id] = el }}
      className={cls(id)}
      aria-hidden={id !== nav.view || undefined}
      inert={id !== nav.view || undefined}
      onAnimationEnd={e => { if (e.target === e.currentTarget && id === nav.from) finish(nav.n) }}
    >
      {child}
    </div>
  )

  const mounted = stories.filter(s => visited.includes(s.id) || s.id === nav.view)
  return (
    <div style={{ '--bt': `${band.bt}px`, '--bb': `${band.bb}px` } as CSSProperties}>
      {layer('start', <StartPage stories={stories} go={go} intro={intro} error={error} />)}
      {mounted.map(s => {
        const Page = PAGES[s.theme]
        return layer(s.id, <Page story={s} stories={stories} go={go} />)
      })}
      {nav.from && ft === 'plates' && vt === 'highlighter' && <div key={nav.n} className="sw-scanbar" aria-hidden="true" />}
    </div>
  )
}
