import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Image,
  InputNumber,
  List,
  message,
  Modal,
  Row,
  Select,
  Skeleton,
  Space,
  Switch,
  Tag,
  Typography,
} from 'antd'
import {
  DownloadOutlined,
  FileZipOutlined,
  TranslationOutlined,
} from '@ant-design/icons'
import { episodeAPI, storyAPI } from '../services/api'
import type { Episode, InlineTranslatedPage, Story } from '../types'

const { Title, Text, Paragraph } = Typography

const LANGUAGE_OPTIONS = [
  { label: 'Tiếng Việt', value: 'vietnamese' },
  { label: 'Tiếng Anh', value: 'english' },
  { label: 'Tiếng Nhật', value: 'japanese' },
  { label: 'Tiếng Hàn', value: 'korean' },
  { label: 'Tiếng Trung', value: 'chinese' },
  { label: 'Tiếng Pháp', value: 'french' },
]

const TranslatePage: React.FC = () => {
  const [form] = Form.useForm()
  const [stories, setStories] = useState<Story[]>([])
  const [storiesLoading, setStoriesLoading] = useState(false)
  const [selectedStory, setSelectedStory] = useState<Story | null>(null)
  const [storyLoading, setStoryLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [availablePages, setAvailablePages] = useState<number[]>([])
  const [pagesLoading, setPagesLoading] = useState(false)
  const [inlinePages, setInlinePages] = useState<InlineTranslatedPage[]>([])
  const [inlineModalVisible, setInlineModalVisible] = useState(false)

  const watchedStoryName = Form.useWatch('storyName', form)
  const watchedChapterNumber = Form.useWatch('chapterNumber', form)
  const watchedResponseMode = Form.useWatch('responseMode', form)
  const watchedForceTranslate = Form.useWatch('forceTranslate', form)

  useEffect(() => {
    const loadStories = async () => {
      setStoriesLoading(true)
      try {
        const response = await storyAPI.getAll()
        if (response.data) {
          setStories(response.data)
        }
      } catch (error) {
        console.error(error)
        message.error('Không thể tải danh sách truyện')
      } finally {
        setStoriesLoading(false)
      }
    }
    loadStories()
  }, [])

  const handleStoryChange = async (storyName?: string) => {
    form.setFieldsValue({ chapterNumber: undefined })
    setSelectedStory(null)
    if (!storyName) {
      return
    }
    setStoryLoading(true)
    try {
      const response = await storyAPI.getByName(storyName)
      if (response.data) {
        setSelectedStory(response.data)
        const latestEpisode = response.data.episodes?.sort(
          (a, b) => (b.chapter_number || 0) - (a.chapter_number || 0)
        )[0]
        form.setFieldsValue({ chapterNumber: latestEpisode?.chapter_number })
      }
    } catch (error) {
      console.error(error)
      message.error('Không thể tải thông tin truyện')
    } finally {
      setStoryLoading(false)
    }
  }

  const handleTranslate = async (values: {
    storyName: string
    chapterNumber: number
    targetLanguage: string
    responseMode: 'zip' | 'inline'
    forceTranslate: boolean
    selectedPages?: number[]
  }) => {
    const { storyName, chapterNumber, targetLanguage, selectedPages, responseMode, forceTranslate } = values
    setDownloading(true)
    try {
      const result = await episodeAPI.translate(
        storyName,
        chapterNumber,
        targetLanguage,
        selectedPages && selectedPages.length > 0 ? selectedPages : undefined,
        responseMode,
        forceTranslate
      )

      if (responseMode === 'zip') {
        const blob = result as Blob
        const fileName = `translated_${storyName.replace(/\s+/g, '_')}_chapter_${chapterNumber}.zip`

        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = fileName
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)

        message.success('Đã tạo gói ảnh dịch, vui lòng kiểm tra file tải về')
      } else {
        let inlineResult: { mode: 'inline'; pages: InlineTranslatedPage[] }
        if (result instanceof Blob) {
          const text = await result.text()
          inlineResult = JSON.parse(text)
        } else {
          inlineResult = result as { mode: 'inline'; pages: InlineTranslatedPage[] }
        }
        setInlinePages(inlineResult.pages ?? [])
        setInlineModalVisible(true)
        message.success('Đã nhận dữ liệu dịch, vui lòng xem bên dưới')
      }
    } catch (error) {
      console.error(error)
      message.error('Không thể dịch tập truyện, vui lòng thử lại sau')
    } finally {
      setDownloading(false)
    }
  }

  const sortedEpisodes: Episode[] = useMemo(() => {
    if (!selectedStory?.episodes) return []
    return [...selectedStory.episodes].sort(
      (a, b) => (b.chapter_number || 0) - (a.chapter_number || 0)
    )
  }, [selectedStory?.episodes])

  useEffect(() => {
    const fetchPages = async () => {
      if (!watchedStoryName || watchedChapterNumber === undefined || watchedChapterNumber === null) {
        setAvailablePages([])
        form.setFieldsValue({ selectedPages: undefined })
        return
      }
      setPagesLoading(true)
      try {
        const response = await episodeAPI.getDetail(watchedStoryName, watchedChapterNumber)
        const pages = response.data?.pages?.map((page) => page.page_number).filter((num): num is number => typeof num === 'number') ?? []
        setAvailablePages(pages)
        form.setFieldsValue({ selectedPages: undefined })
      } catch (error) {
        console.error(error)
        setAvailablePages([])
        message.warning('Không thể tải danh sách trang, sẽ dịch toàn bộ tập.')
      } finally {
        setPagesLoading(false)
      }
    }
    fetchPages()
  }, [form, watchedStoryName, watchedChapterNumber])

  const pageOptions = useMemo(
    () => availablePages.map((pageNumber) => ({ label: `Trang ${pageNumber}`, value: pageNumber })),
    [availablePages]
  )

  const formatDate = (value?: string) => {
    if (!value) return 'N/A'
    return new Date(value).toLocaleString('vi-VN')
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <TranslationOutlined className="text-3xl" />
        <div>
          <Title level={3} className="mb-0">
            Dịch truyện nhanh
          </Title>
          <Text type="secondary">
            Chọn truyện, tập và ngôn ngữ đích để nhận gói ảnh đã được dịch và inpaint.
          </Text>
        </div>
      </div>

      <Alert
        message="Lưu ý"
        description="Tập truyện cần được xử lý transcript trước khi dịch. Kết quả dịch trả về là tệp ZIP chứa các trang đã được inpaint."
        type="info"
        showIcon
      />

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={14}>
          <Card title="Thiết lập dịch truyện">
            <Form
              layout="vertical"
              form={form}
              onFinish={handleTranslate}
              initialValues={{ targetLanguage: 'vietnamese', responseMode: 'zip', forceTranslate: false }}
            >
              <Form.Item
                label="Truyện"
                name="storyName"
                rules={[{ required: true, message: 'Vui lòng chọn truyện' }]}
              >
                <Select
                  placeholder="Chọn truyện cần dịch"
                  loading={storiesLoading}
                  showSearch
                  allowClear
                  optionFilterProp="children"
                  onChange={(value) => handleStoryChange(value)}
                >
                  {stories.map((story) => (
                    <Select.Option key={story.story_name} value={story.story_name}>
                      {story.story_name}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="Số tập"
                    name="chapterNumber"
                    rules={[{ required: true, message: 'Vui lòng nhập số tập' }]}
                  >
                    <InputNumber min={1} className="w-full" placeholder="Nhập số tập" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="Ngôn ngữ đích"
                    name="targetLanguage"
                    rules={[{ required: true, message: 'Vui lòng chọn ngôn ngữ đích' }]}
                  >
                    <Select options={LANGUAGE_OPTIONS} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                label="Trang cần dịch (tuỳ chọn)"
                name="selectedPages"
                extra="Bỏ trống để dịch toàn bộ tập"
              >
                <Select
                  mode="multiple"
                  placeholder={
                    pagesLoading
                      ? 'Đang tải danh sách trang...'
                      : availablePages.length
                        ? 'Chọn một hoặc nhiều trang'
                        : 'Chưa có danh sách trang, sẽ dịch toàn bộ'
                  }
                  loading={pagesLoading}
                  options={pageOptions}
                  disabled={pagesLoading || availablePages.length === 0}
                  allowClear
                  showSearch
                />
              </Form.Item>

              <Form.Item
                label="Kiểu trả về"
                name="responseMode"
                rules={[{ required: true, message: 'Vui lòng chọn kiểu trả về' }]}
              >
                <Select
                  options={[
                    { label: 'Tải file ZIP', value: 'zip' },
                    { label: 'Hiển thị trực tiếp', value: 'inline' },
                  ]}
                />
              </Form.Item>

              <Form.Item
                label="Luôn dịch lại"
                name="forceTranslate"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Divider />

              <Space>
                <Button
                  type="primary"
                  icon={<DownloadOutlined />}
                  loading={downloading}
                  htmlType="submit"
                >
                  {watchedResponseMode === 'inline' ? 'Dịch và hiển thị' : 'Dịch và tải về'}
                </Button>
                <Button
                  htmlType="reset"
                  onClick={() => {
                    form.resetFields()
                    setSelectedStory(null)
                    form.setFieldsValue({ targetLanguage: 'vietnamese', forceTranslate: false, responseMode: 'zip' })
                    setInlinePages([])
                    setInlineModalVisible(false)
                  }}
                >
                  Đặt lại
                </Button>
              </Space>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card title="Thông tin truyện" className="min-h-[320px]">
            {storyLoading ? (
              <Skeleton active paragraph={{ rows: 6 }} />
            ) : !selectedStory ? (
              <div className="flex flex-col items-center justify-center py-10 text-center text-gray-500">
                <FileZipOutlined className="text-4xl mb-3" />
                <Paragraph>Chọn một truyện để xem danh sách tập có sẵn.</Paragraph>
              </div>
            ) : (
              <Space direction="vertical" size="large" className="w-full">
                <div>
                  <Title level={5}>{selectedStory.story_name}</Title>
                  <Space wrap>
                    <Tag color="purple">Loại: {typeof selectedStory.story_type === 'string' ? selectedStory.story_type : selectedStory.story_type.value}</Tag>
                    <Tag color="green">Nguồn: {typeof selectedStory.source_language === 'string' ? selectedStory.source_language : selectedStory.source_language.value}</Tag>
                    <Tag color="blue">{selectedStory.characters?.length ?? 0} nhân vật</Tag>
                    <Tag color="orange">{selectedStory.episodes?.length ?? 0} tập</Tag>
                  </Space>
                  <Text type="secondary">
                    Cập nhật lần cuối: {formatDate(selectedStory.updated_at)}
                  </Text>
                </div>

                <Divider className="my-2" />

                <div>
                  <Title level={5}>Danh sách tập đã có transcript</Title>
                  {sortedEpisodes.length === 0 ? (
                    <Text type="secondary">Chưa có tập nào được khởi tạo.</Text>
                  ) : (
                    <List
                      dataSource={sortedEpisodes}
                      className="max-h-72 overflow-auto"
                      renderItem={(episode) => (
                        <List.Item
                          key={episode.id ?? episode.chapter_number}
                          actions={[
                            <Button
                              key="select"
                              size="small"
                              type="link"
                              onClick={() => form.setFieldsValue({ chapterNumber: episode.chapter_number })}
                            >
                              Chọn tập này
                            </Button>,
                          ]}
                        >
                          <List.Item.Meta
                            title={`Tập ${episode.chapter_number}`}
                            description={
                              <Text type="secondary">
                                Tạo: {formatDate(episode.created_at)} · Cập nhật: {formatDate(episode.updated_at)}
                              </Text>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  )}
                </div>
              </Space>
            )}
          </Card>
        </Col>
      </Row>
      <Modal
        title="Xem nhanh ảnh dịch"
        open={inlineModalVisible}
        onCancel={() => setInlineModalVisible(false)}
        footer={null}
        width={1000}
      >
        {inlinePages.length === 0 ? (
          <Text>Chưa có dữ liệu để hiển thị.</Text>
        ) : (
          <Space direction="vertical" size="large" className="w-full">
            {inlinePages.map((page) => (
              <div key={page.page_name}>
                <Text strong>{page.page_name}</Text>
                <Row gutter={16} className="mt-2">
                  <Col span={12}>
                    <Image src={page.original} alt={`${page.page_name}-original`} />
                  </Col>
                  <Col span={12}>
                    <Image src={page.translated} alt={`${page.page_name}-translated`} />
                  </Col>
                </Row>
              </div>
            ))}
          </Space>
        )}
      </Modal>
    </div>
  )
}

export default TranslatePage

