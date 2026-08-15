import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, Button, Empty, Spin, message } from 'antd'
import { PlusOutlined, BookOutlined } from '@ant-design/icons'
import { storyAPI } from '../services/api'
import { Story } from '../types'
import CreateStoryModal from '../components/CreateStoryModal'

const StoryList: React.FC = () => {
  const [stories, setStories] = useState<Story[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    loadStories()
  }, [])

  const loadStories = async () => {
    setLoading(true)
    try {
      const response = await storyAPI.getAll()
      if (response && response.data && Array.isArray(response.data)) {
        setStories(response.data)
      } else {
        setStories([])
      }
    } catch (error) {
      message.error('Không thể tải danh sách truyện')
      console.error('Error loading stories:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateSuccess = () => {
    setModalVisible(false)
    loadStories()
    message.success('Tạo truyện thành công!')
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Quản lý Truyện</h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          size="large"
          onClick={() => setModalVisible(true)}
          className="bg-blue-600 hover:bg-blue-700"
        >
          Thêm Truyện Mới
        </Button>
      </div>

      {stories.length === 0 ? (
        <Empty
          description="Chưa có truyện nào"
          className="mt-20"
        >
        </Empty>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {stories.map((story) => (
            <Card
              key={story.id || story.story_name}
              hoverable
              className="shadow-md hover:shadow-lg transition-shadow"
              onClick={() => navigate(`/story/${story.story_name}`)}
            >
              <div className="flex items-start gap-4">
                <div className="text-4xl">
                  <BookOutlined />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold mb-2 text-gray-800">
                    {story.story_name}
                  </h3>
                  <div className="space-y-1 text-sm text-gray-600">
                    <p>
                      <span className="font-medium">Loại:</span>{' '}
                      {typeof story.story_type === 'object' 
                        ? story.story_type.value 
                        : story.story_type}
                    </p>
                    <p>
                      <span className="font-medium">Ngôn ngữ:</span>{' '}
                      {typeof story.source_language === 'object' 
                        ? story.source_language.value 
                        : story.source_language}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      <CreateStoryModal
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        onSuccess={handleCreateSuccess}
      />
    </div>
  )
}

export default StoryList

