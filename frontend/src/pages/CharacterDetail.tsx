import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Button,
  Descriptions,
  Tabs,
  Spin,
  message,
  Image,
  Typography,
} from 'antd'
import {
  ArrowLeftOutlined,
  UserOutlined,
  ContactsOutlined,
  CalendarOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import AddressMatrixTable from '../components/AddressMatrixTable'
import AddAddressMatrixModal from '../components/AddAddressMatrixModal'
import { characterAPI } from '../services/api'
import { CharacterDetail, Character } from '../types'

const { Text, Title } = Typography

const CharacterDetailPage: React.FC = () => {
  const { storyName, characterName } = useParams<{ storyName: string; characterName: string }>()
  const navigate = useNavigate()
  const [characterDetail, setCharacterDetail] = useState<CharacterDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [characters, setCharacters] = useState<Character[]>([])
  const [addressMatrixModalVisible, setAddressMatrixModalVisible] = useState(false)

  useEffect(() => {
    if (storyName && characterName) {
      loadCharacterDetail()
      loadStoryCharacters()
    }
  }, [storyName, characterName])

  const loadCharacterDetail = async () => {
    if (!storyName || !characterName) return
    setLoading(true)
    try {
      const response = await characterAPI.getDetail(storyName, characterName)
      if (response.data) {
        setCharacterDetail(response.data)
      }
    } catch (error) {
      message.error('Không thể tải thông tin nhân vật')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const loadStoryCharacters = async () => {
    if (!storyName) return
    try {
      const storyResponse = await fetch(`/api/v1/metadata/stories/${storyName}`)
      const storyData = await storyResponse.json()
      if (storyData.data?.characters) {
        setCharacters(storyData.data.characters)
      }
    } catch (error) {
      console.error('Error loading characters:', error)
    }
  }

  const handleAddressMatrixUpdate = () => {
    loadCharacterDetail()
  }

  const handleAddAddressMatrix = () => {
    setAddressMatrixModalVisible(true)
  }

  const handleAddressMatrixModalSuccess = () => {
    setAddressMatrixModalVisible(false)
    loadCharacterDetail()
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString('vi-VN')
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Spin size="large" />
      </div>
    )
  }

  if (!characterDetail) {
    return (
      <div className="text-center py-8">
        <Text type="danger">Không tìm thấy nhân vật</Text>
      </div>
    )
  }

  const { character, address_matrix } = characterDetail
  const addressMatrixEntries = Object.entries(address_matrix)

  const tabItems = [
    {
      key: 'info',
      label: (
        <span>
          <UserOutlined />
          Thông Tin
        </span>
      ),
      children: (
        <div>
          <Card>
            <Descriptions title="Thông Tin Nhân Vật" bordered column={{ xxl: 2, xl: 2, lg: 2, md: 1, sm: 1, xs: 1 }}>
              <Descriptions.Item label="Tên Nhân Vật">
                <Text strong>{character.source_name}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Mô Tả">
                {character.description || 'Chưa có mô tả'}
              </Descriptions.Item>
              {character.face && (
                <Descriptions.Item label="Khuôn Mặt">
                  {character.face}
                </Descriptions.Item>
              )}
              {character.hair && (
                <Descriptions.Item label="Tóc">
                  {character.hair}
                </Descriptions.Item>
              )}
              {character.eyes && (
                <Descriptions.Item label="Mắt">
                  {character.eyes}
                </Descriptions.Item>
              )}
              {character.outfit && (
                <Descriptions.Item label="Trang Phục">
                  {character.outfit}
                </Descriptions.Item>
              )}
              {character.accessories && (
                <Descriptions.Item label="Phụ Kiện">
                  {character.accessories}
                </Descriptions.Item>
              )}
              {character.distinctive_features && (
                <Descriptions.Item label="Đặc Điểm Nổi Bật">
                  {character.distinctive_features}
                </Descriptions.Item>
              )}
              <Descriptions.Item label={<><CalendarOutlined /> Ngày Tạo</>}>
                {formatDate(character.created_at)}
              </Descriptions.Item>
              <Descriptions.Item label={<><CalendarOutlined /> Ngày Cập Nhật</>}>
                {formatDate(character.updated_at)}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </div>
      ),
    },
    {
      key: 'address-matrix',
      label: (
        <span>
          <ContactsOutlined />
          Ma Trận Xưng Hô ({addressMatrixEntries.length})
        </span>
      ),
      children: (
        <div>
          <div className="mb-4">
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleAddAddressMatrix}
            >
              Thêm Ma Trận Xưng Hô
            </Button>
          </div>
          <AddressMatrixTable
            storyName={storyName || ''}
            character={character}
            characters={characters}
            addressMatrix={address_matrix}
            onUpdate={handleAddressMatrixUpdate}
          />
        </div>
      ),
    },
  ]

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(`/story/${storyName}`)}
          className="mb-4"
        >
          Quay Lại
        </Button>
        
        <Card>
          <div className="flex gap-6">
            <div className="flex-shrink-0">
              {character.image_path ? (
                <Image
                  src={character.image_path}
                  alt={character.source_name}
                  width={200}
                  height={200}
                  className="object-cover rounded"
                />
              ) : (
                <div className="w-48 h-48 bg-gray-100 flex items-center justify-center rounded">
                  <UserOutlined className="text-6xl text-gray-400" />
                </div>
              )}
            </div>
            <div className="flex-1">
              <Title level={2}>{character.source_name}</Title>
              {character.description && (
                <Text className="text-lg">{character.description}</Text>
              )}
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Tabs items={tabItems} />
      </Card>

      <AddAddressMatrixModal
        visible={addressMatrixModalVisible}
        storyName={storyName || ''}
        character={character}
        characters={characters}
        onCancel={() => setAddressMatrixModalVisible(false)}
        onSuccess={handleAddressMatrixModalSuccess}
      />
    </div>
  )
}

export default CharacterDetailPage

