// Page details for each story the API lists (GET /api/events). The API owns ids, ranges and counts;
// these are the words and moments each page opens with. Preset questions live in data/presets.json.
import type { Preset, Story } from './api'
import presets from '../../data/presets.json'

const PRESETS = presets as Record<string, Preset[]>

type Meta = Partial<Omit<Story, 'id' | 'sample' | 'days' | 'hours'>>

export const STORY_META: Record<string, Meta> = {
  'titanic-1912': {
    title: 'The Titanic',
    theme: 'plates',
    dates_label: 'April 10 to 20, 1912',
    hook: 'Monday’s papers said everyone was saved.',
    question: 'What happened to the Titanic’s passengers?',
    default_cutoff: '1912-04-15',
  },
  'lehman-2008': {
    title: 'Lehman Brothers',
    theme: 'highlighter',
    dates_label: 'September 10 to 19, 2008',
    hook: 'Friday’s reports expected a rescue.',
    question: 'Will Lehman Brothers be rescued?',
    default_cutoff: '2008-09-12',
  },
  'titan-2023': {
    title: 'The Titan submersible',
    theme: 'abyss',
    dates_label: 'June 18 to 23, 2023',
    hook: 'For days, the news was about banging sounds and hope.',
    question: 'What happened to the Titan submersible?',
    // The rail runs in US Eastern time (UTC-4 in June); the API takes any ISO time with an offset.
    start: '2023-06-18T00:00:00-04:00',
    end: '2023-06-23T19:59:00-04:00',
    utc_offset: -4,
    timezone_label: 'Eastern time',
    default_cutoff: '2023-06-21T14:00:00-04:00',
    stops: [
      { at: '2023-06-21T02:00:00-04:00', label: 'Wed 02:00' },
      { at: '2023-06-21T14:00:00-04:00', label: 'Wed 14:00' },
      { at: '2023-06-22T08:00:00-04:00', label: 'Thu 08:00' },
      { at: '2023-06-22T14:00:00-04:00', label: 'Thu 14:00' },
    ],
    markers: [{ at: '2023-06-18T09:45:00-04:00', label: 'Contact lost, 09:45' }],
  },
}

for (const [id, list] of Object.entries(PRESETS)) if (STORY_META[id]) STORY_META[id].presets = list
