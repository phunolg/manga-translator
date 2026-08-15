import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Button,
  Card,
  Collapse,
  Empty,
  Image,
  List,
  Spin,
  Tag,
  Typography,
  message,
} from 'antd'
import { ArrowLeftOutlined, BookOutlined, FileImageOutlined } from '@ant-design/icons'
import { episodeAPI } from '../services/api'
import type { EpisodeDetail, EpisodePageDetail, TranscriptLineItem } from '../types'

const { Text, Title, Paragraph } = Typography

const EpisodeDetail: React.FC = () => {
  const navigate = useNavigate()
  const { storyName, chapterNumber } = useParams<{ storyName: string; chapterNumber: string }>()
  const [episode, setEpisode] = useState<EpisodeDetail | null>(null)
  const [loading, setLoading] = useState(false)

  const parsedChapter = useMemo(() => {
    if (!chapterNumber) return undefined
    const value = Number(chapterNumber)
    return Number.isNaN(value) ? undefined : value
  }, [chapterNumber])

  useEffect(() => {
    const fetchEpisodeDetail = async () => {
      if (!storyName || parsedChapter === undefined) {
        return
      }
      setLoading(true)
      try {
        const response = await episodeAPI.getDetail(storyName, parsedChapter)
        if (response.data) {
          setEpisode(response.data)
        } else {
          setEpisode(null)
          message.info('Không có dữ liệu cho tập này')
        }
      } catch (error) {
        console.error(error)
        message.error('Không thể tải chi tiết tập truyện')
      } finally {
        setLoading(false)
      }
    }

    fetchEpisodeDetail()
  }, [storyName, parsedChapter])

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <Spin size="large" />
      </div>
    )
  }

  if (!episode) {
    return (
      <div className="space-y-4">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(`/story/${encodeURIComponent(storyName ?? '')}`)}
        >
          Quay Lại Truyện
        </Button>
        <Card>
          <Empty description="Không tìm thấy dữ liệu tập truyện" />
        </Card>
      </div>
    )
  }

  const formatDateTime = (value?: string) => {
    if (!value) return 'N/A'
    return new Date(value).toLocaleString('vi-VN')
  }

  const renderTranscript = (transcript: TranscriptLineItem) => (
    <List.Item>
      <div className="w-full space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <Tag color="blue">Dòng {transcript.line_index + 1}</Tag>
          {transcript.speaker && <Tag color="purple">{transcript.speaker}</Tag>}
          {transcript.target && <Tag color="magenta">→ {transcript.target}</Tag>}
          {transcript.text_speech_type && <Tag color="gold">{transcript.text_speech_type}</Tag>}
        </div>
        {transcript.text && (
          <Paragraph className="mb-1">
            <Text strong>Nội dung:</Text> {transcript.text}
          </Paragraph>
        )}
        {transcript.translation && (
          <Paragraph className="mb-1" type="secondary">
            <Text strong>Bản dịch:</Text> {transcript.translation}
          </Paragraph>
        )}
        {transcript.bbox !== undefined && transcript.bbox !== null && (
          <Paragraph className="mb-0 text-xs text-gray-500">
            <Text strong>Bounding Box:</Text> {JSON.stringify(transcript.bbox)}
          </Paragraph>
        )}
      </div>
    </List.Item>
  )

  const collapseItems = (episode.pages || []).map((page: EpisodePageDetail) => ({
    key: String(page.id),
    label: (
      <div className="flex items-center gap-2">
        <FileImageOutlined />
        <span>Trang {page.page_number}</span>
        <Tag color="cyan">{page.transcripts.length} thoại</Tag>
      </div>
    ),
    children: (
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <div className="flex justify-center items-start">
          {page.image_url ? (
            <Image
              src={page.image_url}
              alt={`Trang ${page.page_number}`}
              className="w-full max-h-[1500px] object-contain rounded-md shadow"
              placeholder
            />
          ) : (
            <Card className="w-full flex items-center justify-center" bordered={false}>
              <Empty description="Không có ảnh cho trang này" />
            </Card>
          )}
        </div>
        <div className="space-y-4">
          {page.prose && (
            <Card size="small" title="Prose">
              <Paragraph>{page.prose}</Paragraph>
            </Card>
          )}
          <Card size="small" title="Danh sách thoại">
            {page.transcripts.length === 0 ? (
              <Empty description="Không có thoại" />
            ) : (
              <List
                dataSource={page.transcripts}
                renderItem={renderTranscript}
                itemLayout="vertical"
              />
            )}
          </Card>
        </div>
      </div>
    ),
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate(`/story/${encodeURIComponent(storyName ?? episode.story_name)}`)}
        >
          Quay Lại Truyện
        </Button>
        <Tag icon={<BookOutlined />} color="orange">
          {`Tập ${episode.chapter_number}`}
        </Tag>
      </div>

      <Card>
        <Title level={4} className="mb-4">
          {episode.story_name} - Tập {episode.chapter_number}
        </Title>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <Text strong>Mã tập:</Text> <Text>{episode.id}</Text>
          </div>
          <div>
            <Text strong>Tổng số trang:</Text> <Text>{episode.pages.length}</Text>
          </div>
          <div>
            <Text strong>Ngày tạo:</Text> <Text>{formatDateTime(episode.created_at)}</Text>
          </div>
          <div>
            <Text strong>Ngày cập nhật:</Text> <Text>{formatDateTime(episode.updated_at)}</Text>
          </div>
        </div>
      </Card>

      <Card title="Chi tiết các trang">
        {collapseItems.length === 0 ? (
          <Empty description="Chưa có trang nào" />
        ) : (
          <Collapse items={collapseItems} accordion defaultActiveKey={collapseItems[0]?.key} />
        )}
      </Card>
    </div>
  )
}

export default EpisodeDetail
