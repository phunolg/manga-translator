export interface Episode {
  id: number
  chapter_number: number
  created_at?: string
  updated_at?: string
}

export interface TranscriptLineItem {
  line_index: number
  speaker?: string | null
  text?: string | null
  text_speech_type?: string | null
  target?: string | null
  translation?: string | null
  bbox?: unknown
}

export interface EpisodePageDetail {
  id: number
  page_number: number
  prose?: string | null
  image_url?: string | null
  transcripts: TranscriptLineItem[]
}

export interface EpisodeDetail {
  id: number
  story_id: number
  story_name: string
  chapter_number: number
  created_at?: string
  updated_at?: string
  pages: EpisodePageDetail[]
}

export interface TranslateDictItem {
  language: string
  dictionary: Record<string, string>
}

export interface Story {
  id?: number
  story_name: string
  story_type: string | { value: string }
  source_language: string | { value: string }
  created_at?: string
  updated_at?: string
  characters?: Character[]
  episodes?: Episode[]
  translate_dicts?: TranslateDictItem[]
  mapping_names?: MappingNameGroup[]
}

export interface Character {
  id: number
  source_name: string
  description?: string
  image_path?: string
  face?: string
  hair?: string
  eyes?: string
  outfit?: string
  accessories?: string
  distinctive_features?: string
  created_at?: string
  updated_at?: string
}

export interface MappingNameGroup {
  language: string
  dictionary: Record<string, string>
}

export interface CharacterDetail {
  character: Character
  address_matrix: AddressMatrix
}

export interface AddressMatrix {
  [targetName: string]: string
}

export interface TranslateDict {
  [key: string]: string
}

export interface CreateTranslateDictRequest {
  language: string
  dictionary: Record<string, string>
}

export interface CreateStoryRequest {
  story_name: string
  story_type: string
  source_language: string
}

export interface InlineTranslatedPage {
  page_name: string
  original: string
  translated: string
}

export interface InlineTranslateResponse {
  mode: 'inline'
  pages: InlineTranslatedPage[]
}

export interface CreateCharacterRequest {
  name_character: string
  description?: string
  character_image?: File
}

export interface UpdateCharacterRequest {
  name?: string
  description?: string
  character_image?: File
}

export interface ApiResponse<T> {
  message: string
  data?: T
  status_code?: number
}

