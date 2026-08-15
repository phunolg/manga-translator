import React, { useState } from 'react'
import { Modal, Form, Input, Upload, message, Button } from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import { characterAPI } from '../services/api'
import { CreateCharacterRequest } from '../types'

interface AddCharacterModalProps {
  visible: boolean
  storyName: string
  onCancel: () => void
  onSuccess: () => void
}

const AddCharacterModal: React.FC<AddCharacterModalProps> = ({
  visible,
  storyName,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      
      if (fileList.length === 0) {
        message.error('Vui lòng chọn ảnh nhân vật')
        return
      }

      setLoading(true)
      
      const characterData: CreateCharacterRequest = {
        name_character: values.name_character,
        description: values.description,
        character_image: fileList[0].originFileObj as File,
      }

      await characterAPI.create(storyName, characterData)
      form.resetFields()
      setFileList([])
      onSuccess()
    } catch (error: any) {
      if (error.errorFields) {
        return
      }
      message.error('Không thể thêm nhân vật')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileChange = ({ fileList: newFileList }: { fileList: UploadFile[] }) => {
    setFileList(newFileList)
  }

  const beforeUpload = () => {
    return false // Prevent auto upload
  }

  return (
    <Modal
      title="Thêm Nhân Vật Mới"
      open={visible}
      onOk={handleSubmit}
      onCancel={onCancel}
      confirmLoading={loading}
      okText="Thêm"
      cancelText="Hủy"
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        className="mt-4"
      >
        <Form.Item
          name="name_character"
          label="Tên Nhân Vật"
          rules={[{ required: true, message: 'Vui lòng nhập tên nhân vật' }]}
        >
          <Input placeholder="Nhập tên nhân vật" />
        </Form.Item>

        <Form.Item
          name="description"
          label="Mô Tả"
        >
          <Input.TextArea
            rows={4}
            placeholder="Nhập mô tả nhân vật (tùy chọn)"
          />
        </Form.Item>

        <Form.Item
          label="Ảnh Nhân Vật"
          required
        >
          <Upload
            fileList={fileList}
            onChange={handleFileChange}
            beforeUpload={beforeUpload}
            maxCount={1}
            accept="image/*"
          >
            <Button icon={<UploadOutlined />}>Chọn Ảnh</Button>
          </Upload>
        </Form.Item>
      </Form>
    </Modal>
  )
}

export default AddCharacterModal

