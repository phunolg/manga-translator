import React, { useState } from 'react'
import { Modal, Form, Input, Select, message } from 'antd'
import { storyAPI } from '../services/api'
import { CreateStoryRequest } from '../types'

interface CreateStoryModalProps {
  visible: boolean
  onCancel: () => void
  onSuccess: () => void
}

const storyTypes = [
  { label: 'Truyện Hiện Đại', value: 'truyện hiện đại' },
  { label: 'Truyện Cổ Trang', value: 'truyện cổ trang' },
  { label: 'Wuxia', value: 'wuxia' },
  { label: 'Xianxia', value: 'xianxia' },
  { label: 'Court Intrigue', value: 'court_intrigue' },
  { label: 'Fantasy General', value: 'fantasy_general' },
]

const languages = [
  { label: 'Tiếng Việt', value: 'vietnamese' },
  { label: 'Tiếng Anh', value: 'english' },
  { label: 'Tiếng Nhật', value: 'japanese' },
  { label: 'Tiếng Hàn', value: 'korean' },
  { label: 'Tiếng Trung', value: 'chinese' },
  { label: 'Tiếng Pháp', value: 'french' },
]

const CreateStoryModal: React.FC<CreateStoryModalProps> = ({
  visible,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      
      const storyData: CreateStoryRequest = {
        story_name: values.story_name,
        story_type: values.story_type,
        source_language: values.source_language,
      }

      await storyAPI.create(storyData)
      form.resetFields()
      onSuccess()
    } catch (error: any) {
      if (error.errorFields) {
        return
      }
      message.error('Không thể tạo truyện')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal
      title="Tạo Truyện Mới"
      open={visible}
      onOk={handleSubmit}
      onCancel={onCancel}
      confirmLoading={loading}
      okText="Tạo"
      cancelText="Hủy"
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        className="mt-4"
      >
        <Form.Item
          name="story_name"
          label="Tên Truyện"
          rules={[{ required: true, message: 'Vui lòng nhập tên truyện' }]}
        >
          <Input placeholder="Nhập tên truyện" />
        </Form.Item>

        <Form.Item
          name="story_type"
          label="Loại Truyện"
          rules={[{ required: true, message: 'Vui lòng chọn loại truyện' }]}
        >
          <Select placeholder="Chọn loại truyện" options={storyTypes} />
        </Form.Item>

        <Form.Item
          name="source_language"
          label="Ngôn Ngữ Gốc"
          rules={[{ required: true, message: 'Vui lòng chọn ngôn ngữ gốc' }]}
        >
          <Select placeholder="Chọn ngôn ngữ gốc" options={languages} />
        </Form.Item>
      </Form>
    </Modal>
  )
}

export default CreateStoryModal

