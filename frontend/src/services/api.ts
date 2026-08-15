import axios, { AxiosInstance } from 'axios'
import {
  Story,
  CreateStoryRequest,
  CreateCharacterRequest,
  UpdateCharacterRequest,
  AddressMatrix,
  ApiResponse,
  Character,
  CreateTranslateDictRequest,
  TranslateDictItem,
  CharacterDetail,
  EpisodeDetail,
  MappingNameGroup,
  InlineTranslateResponse,
} from '../types'

const API_BASE_URL = '/api/v1'

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})


// Stories API
export const storyAPI = {
  // Get all stories
  getAll: async (): Promise<ApiResponse<Story[]>> => {
    const response = await api.get<ApiResponse<Story[]>>('/metadata/stories')
    return response.data
  },
  
  // Create story
  create: async (storyData: CreateStoryRequest): Promise<ApiResponse<Story>> => {
    const response = await api.post<ApiResponse<Story>>('/metadata/stories', storyData)
    return response.data
  },
  
  // Get story by name
  getByName: async (storyName: string): Promise<ApiResponse<Story>> => {
    const response = await api.get<ApiResponse<Story>>(`/metadata/stories/${storyName}`)
    return response.data
  },
  
  // Update story metadata
  updateMetadata: async (
    storyName: string,
    metadata: {
      mapping_name?: Record<string, string>
      translate_dict?: Record<string, Record<string, string>>
      story_type?: string
      source_language?: string
    }
  ): Promise<ApiResponse<Story>> => {
    const formData = new FormData()
    if (metadata.mapping_name) {
      formData.append('mapping_name', JSON.stringify(metadata.mapping_name))
    }
    if (metadata.translate_dict) {
      formData.append('translate_dict', JSON.stringify(metadata.translate_dict))
    }
    if (metadata.story_type) {
      formData.append('story_type', metadata.story_type)
    }
    if (metadata.source_language) {
      formData.append('source_language', metadata.source_language)
    }
    
    const response = await api.patch<ApiResponse<Story>>(
      `/metadata/metadata/${storyName}`,
      formData
    )
    return response.data
  },
  
  // Create translate dictionary
  createTranslateDict: async (
    storyName: string,
    translateDict: CreateTranslateDictRequest
  ): Promise<ApiResponse<TranslateDictItem>> => {
    const response = await api.post<ApiResponse<TranslateDictItem>>(
      `/metadata/stories/${storyName}/translate-dict`,
      translateDict
    )
    return response.data
  },

  getMappingNames: async (
    storyName: string,
    language?: string
  ): Promise<ApiResponse<MappingNameGroup[]>> => {
    const response = await api.get<ApiResponse<MappingNameGroup[]>>(
      `/metadata/stories/${storyName}/mapping-names`,
      { params: { language } }
    )
    return response.data
  },

  saveMappingNames: async (
    storyName: string,
    payload: { language: string; dictionary: Record<string, string> }
  ): Promise<ApiResponse<MappingNameGroup>> => {
    const response = await api.post<ApiResponse<MappingNameGroup>>(
      `/metadata/stories/${storyName}/mapping-names`,
      payload
    )
    return response.data
  },

  deleteMappingName: async (
    storyName: string,
    language: string,
    source: string
  ): Promise<ApiResponse<void>> => {
    const response = await api.delete<ApiResponse<void>>(
      `/metadata/stories/${storyName}/mapping-names`,
      { params: { language, source } }
    )
    return response.data
  },
}

// Characters API
export const characterAPI = {
  // Create character
  create: async (
    storyName: string,
    characterData: CreateCharacterRequest
  ): Promise<ApiResponse<Character>> => {
    const formData = new FormData()
    formData.append('story_name', storyName)
    formData.append('name_character', characterData.name_character)
    if (characterData.description) {
      formData.append('description', characterData.description)
    }
    if (characterData.character_image) {
      formData.append('character_image', characterData.character_image)
    }
    
    const response = await api.post<ApiResponse<Character>>('/character', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
  
  // Get character detail
  getDetail: async (
    storyName: string,
    characterName: string
  ): Promise<ApiResponse<CharacterDetail>> => {
    const response = await api.get<ApiResponse<CharacterDetail>>(
      `/character/${storyName}/${characterName}`
    )
    return response.data
  },
  
  // Update character
  update: async (
    storyName: string,
    characterName: string,
    characterData: UpdateCharacterRequest
  ): Promise<ApiResponse<Character>> => {
    const formData = new FormData()
    if (characterData.name) {
      formData.append('name', characterData.name)
    }
    if (characterData.description !== undefined) {
      formData.append('description', characterData.description)
    }
    if (characterData.character_image) {
      formData.append('character_image', characterData.character_image)
    }
    
    const response = await api.patch<ApiResponse<Character>>(
      `/character/${storyName}/${characterName}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  },
}

// Address Matrix API
export const addressMatrixAPI = {
  // Get address matrices for a character
  get: async (
    storyName: string,
    characterName: string
  ): Promise<ApiResponse<{ address_matrix: AddressMatrix }>> => {
    const response = await api.get<ApiResponse<{ address_matrix: AddressMatrix }>>(
      `/character/${storyName}/${characterName}/address-matrix`
    )
    return response.data
  },
  
  // Merge address matrices (bổ sung/ghi đè)
  merge: async (
    storyName: string,
    characterName: string,
    addressMatrix: AddressMatrix
  ): Promise<ApiResponse<{ address_matrix: AddressMatrix }>> => {
    const response = await api.post<ApiResponse<{ address_matrix: AddressMatrix }>>(
      `/character/${storyName}/${characterName}/address-matrix`,
      addressMatrix
    )
    return response.data
  },
  
  // Replace all address matrices (ghi đè toàn bộ)
  replace: async (
    storyName: string,
    characterName: string,
    addressMatrix: AddressMatrix
  ): Promise<ApiResponse<{ address_matrix: AddressMatrix }>> => {
    const response = await api.put<ApiResponse<{ address_matrix: AddressMatrix }>>(
      `/character/${storyName}/${characterName}/address-matrix`,
      addressMatrix
    )
    return response.data
  },
  
  // Delete address matrix
  delete: async (
    storyName: string,
    characterName: string,
    targetName: string
  ): Promise<ApiResponse<void>> => {
    const response = await api.delete<ApiResponse<void>>(
      `/character/${storyName}/${characterName}/address-matrix/${targetName}`
    )
    return response.data
  },
}

// Episode API
export const episodeAPI = {
  create: async (
    storyName: string,
    chapterNumber: number,
    chapterPages: File[]
  ): Promise<ApiResponse<any>> => {
    const formData = new FormData()
    formData.append('story_name', storyName)
    formData.append('chapter_number', String(chapterNumber))
    chapterPages.forEach((file) => {
      formData.append('chapter_pages', file)
    })

    const response = await api.post<ApiResponse<any>>('/episode', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  translate: async (
    storyName: string,
    chapterNumber: number,
    language: string,
    pages?: number[],
    mode: 'zip' | 'inline' = 'zip',
    forceTranslate = false
  ): Promise<Blob | InlineTranslateResponse> => {
    const encodedStoryName = encodeURIComponent(storyName)
    const params: Record<string, string | number | number[] | boolean> = { language, mode }
    if (pages && pages.length > 0) {
      params.pages = pages
    }
    if (forceTranslate) {
      params.translate = forceTranslate
    }
    const response = await api.get(
      `/episode/${encodedStoryName}/chapters/${chapterNumber}/translated`,
      {
        params,
        paramsSerializer: {
          indexes: null,
        },
        responseType: mode === 'zip' ? 'blob' : 'json',
      }
    )
    return response.data
  },


  getDetail: async (
    storyName: string,
    chapterNumber: number
  ): Promise<ApiResponse<EpisodeDetail>> => {
    const encodedStoryName = encodeURIComponent(storyName)
    const response = await api.get<ApiResponse<EpisodeDetail>>(
      `/episode/${encodedStoryName}/chapters/${chapterNumber}`
    )
    return response.data
  },
}

export default api

