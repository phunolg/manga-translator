import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Button,
  Descriptions,
  Tabs,
  Spin,
  message,
  Tag,
  List,
  Typography,
  Modal,
  Form,
  InputNumber,
  Upload,
  Popconfirm,
} from 'antd'
import {
  ArrowLeftOutlined,
  UserOutlined,
  TranslationOutlined,
  BookOutlined,
  CalendarOutlined,
  TagsOutlined,
  DeleteOutlined,
} from '@ant-design/icons'
import CharacterList from '../components/CharacterList'
import AddCharacterModal from '../components/AddCharacterModal'
import AddTranslateDictModal from '../components/AddTranslateDictModal'
import AddMappingNameModal from '../components/AddMappingNameModal'
import { storyAPI, episodeAPI } from '../services/api'
import { Story } from '../types'
import type { UploadFile, RcFile } from 'antd/es/upload/interface'

const { Text } = Typography

const StoryDetail: React.FC = () => {
  const { storyName } = useParams<{ storyName: string }>()
  const navigate = useNavigate()
  const [story, setStory] = useState<Story | null>(null)
  const [loading, setLoading] = useState(false)
  const [characterModalVisible, setCharacterModalVisible] = useState(false)
  const [translateDictModalVisible, setTranslateDictModalVisible] = useState(false)
  const [editingLanguage, setEditingLanguage] = useState<string | undefined>(undefined)
  const [mappingModalVisible, setMappingModalVisible] = useState(false)
  const [mappingEditingLanguage, setMappingEditingLanguage] = useState<string | undefined>(undefined)
  const [episodeModalVisible, setEpisodeModalVisible] = useState(false)
  const [episodeSubmitting, setEpisodeSubmitting] = useState(false)
  const [episodeFileList, setEpisodeFileList] = useState<UploadFile[]>([])
  const [episodeForm] = Form.useForm()

  useEffect(() => {
    if (storyName) {
      loadStory()
    }
  }, [storyName])

  const loadStory = async () => {
    if (!storyName) return
    setLoading(true)
    try {
      const response = await storyAPI.getByName(storyName)
      if (response.data) {
        setStory(response.data)
      }
    } catch (error) {
      message.error('Không thể tải thông tin truyện')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const handleAddCharacter = () => {
    setCharacterModalVisible(true)
  }

  const handleAddTranslateDict = () => {
    setEditingLanguage(undefined)
    setTranslateDictModalVisible(true)
  }

  const handleEditTranslateDict = (language: string) => {
    setEditingLanguage(language)
    setTranslateDictModalVisible(true)
  }

  const handleAddMappingName = () => {
    setMappingEditingLanguage(undefined)
    setMappingModalVisible(true)
  }

  const handleEditMappingName = (language: string) => {
    setMappingEditingLanguage(language)
    setMappingModalVisible(true)
  }

  const handleCharacterSuccess = () => {
    setCharacterModalVisible(false)
    loadStory()
    message.success('Thêm nhân vật thành công!')
  }

  const handleTranslateDictSuccess = () => {
    setTranslateDictModalVisible(false)
    loadStory()
    message.success('Thêm từ điển thành công!')
  }

  const handleMappingModalSuccess = () => {
    setMappingModalVisible(false)
    setMappingEditingLanguage(undefined)
    loadStory()
    message.success('Đã lưu mapping name')
  }

  const handleDeleteMappingEntry = async (language: string, source: string) => {
    if (!storyName) return
    try {
      await storyAPI.deleteMappingName(storyName, language, source)
      message.success(`Đã xoá '${source}' khỏi ${language}`)
      loadStory()
    } catch (error) {
      console.error(error)
      message.error('Không thể xoá mapping name')
    }
  }

  const handleOpenEpisodeModal = () => {
    setEpisodeModalVisible(true)
  }

  const handleEpisodeModalCancel = () => {
    setEpisodeModalVisible(false)
    setEpisodeFileList([])
    episodeForm.resetFields()
  }

  const handleEpisodeSubmit = async (values: { chapter_number: number }) => {
    if (!storyName) {
      message.error('Không xác định được tên truyện')
      return
    }
    const files: File[] = episodeFileList
      .map((file) => file.originFileObj)
      .filter((file): file is RcFile => !!file)
      .map((file) => file as File)

    if (files.length === 0) {
      message.error('Vui lòng chọn ít nhất một trang truyện')
      return
    }

    try {
      setEpisodeSubmitting(true)
      await episodeAPI.create(storyName, values.chapter_number, files)
      message.success('Yêu cầu tạo tập đã được đưa vào hàng đợi. Vui lòng tải lại sau ít phút.')
      handleEpisodeModalCancel()
    } catch (error) {
      console.error(error)
      message.error('Không thể tạo tập mới')
    } finally {
      setEpisodeSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Spin size="large" />
      </div>
    )
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString('vi-VN')
  }

  const getStoryTypeValue = (storyType: string | { value: string } | undefined) => {
    if (!storyType) return 'Unknown'
    return typeof storyType === 'string' ? storyType : storyType.value
  }

  const getSourceLanguageValue = (sourceLanguage: string | { value: string } | undefined) => {
    if (!sourceLanguage) return 'Unknown'
    return typeof sourceLanguage === 'string' ? sourceLanguage : sourceLanguage.value
  }

  const characters = story?.characters || []
  const episodes = story?.episodes || []
  const translateDicts = story?.translate_dicts || []
  const mappingNames = story?.mapping_names || []

  const tabItems = [
    {
      key: 'characters',
      label: (
        <span>
          <UserOutlined />
          Nhân Vật ({characters.length})
        </span>
      ),
      children: (
        <div>
          <div className="mb-4">
            <Button
              type="primary"
              onClick={handleAddCharacter}
              className="mb-4"
            >
              Thêm Nhân Vật
            </Button>
          </div>
          <CharacterList
            characters={characters}
            storyName={storyName || ''}
          />
        </div>
      ),
    },
    {
      key: 'episodes',
      label: (
        <span>
          <BookOutlined />
          Tập ({episodes.length})
        </span>
      ),
      children: (
        <div>
              <Button
                type="primary"
                onClick={handleOpenEpisodeModal}
                className="mb-4"
                disabled={!storyName}
              >
                Tạo Tập Mới
              </Button>
          {episodes.length === 0 ? (
            <p className="text-gray-600">Chưa có tập nào</p>
          ) : (
            <List
              dataSource={episodes}
              renderItem={(episode) => (
                <List.Item
                  actions={[
                    <Button
                      type="link"
                      key="view"
                      onClick={() =>
                        navigate(`/story/${encodeURIComponent(storyName || '')}/episode/${episode.chapter_number}`)
                      }
                    >
                      Xem chi tiết
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    title={`Tập ${episode.chapter_number}`}
                    description={
                      <div>
                        <Text type="secondary">
                          Tạo: {formatDate(episode.created_at)} | 
                          Cập nhật: {formatDate(episode.updated_at)}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          )}
        </div>
      ),
    },
    {
      key: 'translate-dict',
      label: (
        <span>
          <TranslationOutlined />
          Từ Điển Dịch ({translateDicts.length})
        </span>
      ),
      children: (
        <div>
          <Button
            type="primary"
            onClick={handleAddTranslateDict}
            className="mb-4"
          >
            Thêm Từ Điển Ngôn Ngữ Mới
          </Button>
          {translateDicts.length === 0 ? (
            <p className="text-gray-600">Chưa có từ điển nào</p>
          ) : (
            <div className="space-y-4">
              {translateDicts.map((dict, index) => (
                <Card
                  key={index}
                  size="small"
                  className="mb-2"
                  actions={[
                    <Button
                      type="link"
                      key="edit"
                      onClick={() => handleEditTranslateDict(dict.language)}
                    >
                      Chỉnh Sửa
                    </Button>,
                  ]}
                >
                  <div className="mb-2">
                    <Tag color="blue">{dict.language}</Tag>
                    <Text type="secondary" className="ml-2">
                      {Object.keys(dict.dictionary).length} từ
                    </Text>
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    <List
                      size="small"
                      dataSource={Object.entries(dict.dictionary)}
                      renderItem={([key, value]) => (
                        <List.Item>
                          <Text strong>{key}:</Text> <Text>{value}</Text>
                        </List.Item>
                      )}
                    />
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'mapping-names',
      label: (
        <span>
          <TagsOutlined />
          Mapping Name ({mappingNames.length})
        </span>
      ),
      children: (
        <div>
          <Button type="primary" onClick={handleAddMappingName} className="mb-4">
            Thêm Mapping Name
          </Button>
          {mappingNames.length === 0 ? (
            <p className="text-gray-600">Chưa có mapping name nào</p>
          ) : (
            <div className="space-y-4">
              {mappingNames.map((group, index) => (
                <Card
                  key={`${group.language}-${index}`}
                  size="small"
                  className="mb-2"
                  actions={[
                    <Button
                      type="link"
                      key="edit"
                      onClick={() => handleEditMappingName(group.language)}
                    >
                      Chỉnh Sửa
                    </Button>,
                  ]}
                >
                  <div className="mb-2">
                    <Tag color="volcano">{group.language}</Tag>
                    <Text type="secondary" className="ml-2">
                      {Object.keys(group.dictionary).length} mục
                    </Text>
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    <List
                      size="small"
                      dataSource={Object.entries(group.dictionary)}
                      renderItem={([source, translation]) => (
                        <List.Item
                          actions={[
                            <Popconfirm
                              key="delete"
                              title="Xoá mapping này?"
                              okText="Xoá"
                              cancelText="Huỷ"
                              onConfirm={() => handleDeleteMappingEntry(group.language, source)}
                            >
                              <Button
                                type="text"
                                danger
                                icon={<DeleteOutlined />}
                                size="small"
                              />
                            </Popconfirm>,
                          ]}
                        >
                          <Text strong>{source}:</Text> <Text>{translation}</Text>
                        </List.Item>
                      )}
                    />
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      ),
    },
  ]

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/')}
          className="mb-4"
        >
          Quay Lại
        </Button>
        <Card>
          <Descriptions title={`Thông Tin Truyện: ${story?.story_name}`} bordered column={{ xxl: 2, xl: 2, lg: 2, md: 1, sm: 1, xs: 1 }}>
            <Descriptions.Item label="Tên Truyện">
              {story?.story_name}
            </Descriptions.Item>
            <Descriptions.Item label="Loại Truyện">
              <Tag color="purple">{getStoryTypeValue(story?.story_type)}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Ngôn Ngữ Gốc">
              <Tag color="green">{getSourceLanguageValue(story?.source_language)}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Số Nhân Vật">
              <Tag color="blue">{characters.length}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Số Tập">
              <Tag color="orange">{episodes.length}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Số Từ Điển">
              <Tag color="cyan">{translateDicts.length}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label={<><CalendarOutlined /> Ngày Tạo</>}>
              {formatDate(story?.created_at)}
            </Descriptions.Item>
            <Descriptions.Item label={<><CalendarOutlined /> Ngày Cập Nhật</>}>
              {formatDate(story?.updated_at)}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </div>

      <Card>
        <Tabs items={tabItems} />
      </Card>

      <AddCharacterModal
        visible={characterModalVisible}
        storyName={storyName || ''}
        onCancel={() => setCharacterModalVisible(false)}
        onSuccess={handleCharacterSuccess}
      />

      <AddTranslateDictModal
        visible={translateDictModalVisible}
        storyName={storyName || ''}
        existingTranslateDicts={translateDicts}
        editingLanguage={editingLanguage}
        onCancel={() => {
          setTranslateDictModalVisible(false)
          setEditingLanguage(undefined)
        }}
        onSuccess={handleTranslateDictSuccess}
      />

      <AddMappingNameModal
        visible={mappingModalVisible}
        storyName={storyName || ''}
        existingMappings={mappingNames}
        editingLanguage={mappingEditingLanguage}
        onCancel={() => {
          setMappingModalVisible(false)
          setMappingEditingLanguage(undefined)
        }}
        onSuccess={handleMappingModalSuccess}
      />

      <Modal
        title="Tạo Tập Mới"
        open={episodeModalVisible}
        onCancel={handleEpisodeModalCancel}
        okText="Tạo Tập"
        onOk={() => episodeForm.submit()}
        confirmLoading={episodeSubmitting}
        destroyOnClose
      >
        <Form
          layout="vertical"
          form={episodeForm}
          onFinish={handleEpisodeSubmit}
        >
          <Form.Item
            label="Số Tập"
            name="chapter_number"
            rules={[
              { required: true, message: 'Vui lòng nhập số tập' },
              { type: 'number', min: 1, message: 'Số tập phải lớn hơn 0' },
            ]}
          >
            <InputNumber className="w-full" min={1} />
          </Form.Item>

          <Form.Item
            label="Trang Truyện"
            required
            validateStatus={episodeFileList.length === 0 ? 'error' : undefined}
            help={episodeFileList.length === 0 ? 'Vui lòng chọn ít nhất một trang' : undefined}
          >
            <Upload
              multiple
              fileList={episodeFileList}
              beforeUpload={() => false}
              accept=".png,.jpg,.jpeg,.webp"
              onChange={({ fileList }) => setEpisodeFileList(fileList)}
              listType="picture"
            >
              <Button>Tải lên trang truyện</Button>
            </Upload>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default StoryDetail

