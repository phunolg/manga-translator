import React, { useEffect, useState } from 'react'
import { Modal, Form, Select, Input, Button, message, Alert } from 'antd'
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { storyAPI } from '../services/api'
import { MappingNameGroup } from '../types'

interface AddMappingNameModalProps {
  visible: boolean
  storyName: string
  existingMappings?: MappingNameGroup[]
  editingLanguage?: string
  onCancel: () => void
  onSuccess: () => void
}

const languages = [
  { label: 'Tiếng Việt', value: 'vietnamese' },
  { label: 'Tiếng Anh', value: 'english' },
  { label: 'Tiếng Nhật', value: 'japanese' },
  { label: 'Tiếng Hàn', value: 'korean' },
  { label: 'Tiếng Trung', value: 'chinese' },
  { label: 'Tiếng Pháp', value: 'french' },
]

const AddMappingNameModal: React.FC<AddMappingNameModalProps> = ({
  visible,
  storyName,
  existingMappings = [],
  editingLanguage,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [entries, setEntries] = useState<Array<{ source: string; translation: string }>>([
    { source: '', translation: '' },
  ])
  const [selectedLanguage, setSelectedLanguage] = useState<string | undefined>(editingLanguage)

  useEffect(() => {
    if (visible) {
      if (editingLanguage) {
        const existing = existingMappings.find((dict) => dict.language === editingLanguage)
        if (existing) {
          const data = Object.entries(existing.dictionary).map(([source, translation]) => ({
            source,
            translation,
          }))
          setEntries(data.length > 0 ? data : [{ source: '', translation: '' }])
          form.setFieldsValue({ language: editingLanguage })
          setSelectedLanguage(editingLanguage)
        }
      } else {
        form.resetFields()
        setEntries([{ source: '', translation: '' }])
        setSelectedLanguage(undefined)
      }
    }
  }, [visible, editingLanguage, existingMappings, form])

  const existingLanguages = existingMappings.map((dict) => dict.language)
  const isEditing = !!editingLanguage
  const isLanguageExists =
    selectedLanguage && existingLanguages.includes(selectedLanguage) && !isEditing

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      const language = values.language
      const dictionary: Record<string, string> = {}
      entries.forEach((entry) => {
        if (entry.source && entry.translation) {
          dictionary[entry.source] = entry.translation
        }
      })

      if (Object.keys(dictionary).length === 0) {
        message.error('Vui lòng thêm ít nhất một mapping name')
        return
      }

      setLoading(true)
      await storyAPI.saveMappingNames(storyName, {
        language,
        dictionary,
      })
      message.success(
        isEditing ? 'Cập nhật mapping name thành công!' : 'Thêm mapping name thành công!'
      )
      form.resetFields()
      setEntries([{ source: '', translation: '' }])
      setSelectedLanguage(undefined)
      onSuccess()
    } catch (error: any) {
      if (error?.errorFields) {
        return
      }
      console.error(error)
      message.error('Không thể lưu mapping name')
    } finally {
      setLoading(false)
    }
  }

  const addEntry = () => {
    setEntries([...entries, { source: '', translation: '' }])
  }

  const removeEntry = (index: number) => {
    if (entries.length > 1) {
      setEntries(entries.filter((_, i) => i !== index))
    }
  }

  const updateEntry = (index: number, field: 'source' | 'translation', value: string) => {
    const next = [...entries]
    next[index][field] = value
    setEntries(next)
    form.setFieldsValue({
      [`source_${index}`]: next[index].source,
      [`translation_${index}`]: next[index].translation,
    })
  }

  const handleLanguageChange = (value: string) => {
    setSelectedLanguage(value)
    const existing = existingMappings.find((dict) => dict.language === value)
    if (existing && !isEditing) {
      const data = Object.entries(existing.dictionary).map(([source, translation]) => ({
        source,
        translation,
      }))
      setEntries(data.length > 0 ? data : [{ source: '', translation: '' }])
    } else if (!existing) {
      setEntries([{ source: '', translation: '' }])
    }
  }

  return (
    <Modal
      title={isEditing ? 'Chỉnh Sửa Mapping Name' : 'Thêm Mapping Name'}
      open={visible}
      onOk={handleSubmit}
      onCancel={() => {
        form.resetFields()
        setEntries([{ source: '', translation: '' }])
        setSelectedLanguage(undefined)
        onCancel()
      }}
      confirmLoading={loading}
      okText={isEditing ? 'Cập Nhật' : 'Thêm'}
      cancelText="Hủy"
      width={700}
    >
      <Form form={form} layout="vertical" className="mt-4">
        <Form.Item
          name="language"
          label="Ngôn Ngữ"
          rules={[{ required: true, message: 'Vui lòng chọn ngôn ngữ' }]}
        >
          <Select
            placeholder="Chọn ngôn ngữ"
            options={languages}
            disabled={isEditing}
            onChange={handleLanguageChange}
          />
        </Form.Item>

        {isLanguageExists && (
          <Alert
            message="Ngôn ngữ này đã có mapping name"
            description="Các tên mới sẽ được thêm vào nhóm hiện có. Tên trùng sẽ được cập nhật."
            type="info"
            showIcon
            className="mb-4"
          />
        )}

        <div className="space-y-4">
          <div className="font-medium mb-2">Danh sách tên:</div>
          {entries.map((entry, index) => (
            <div key={index} className="flex gap-2 items-start">
              <Form.Item
                name={`source_${index}`}
                label={index === 0 ? 'Tên gốc' : ''}
                className="flex-1"
                rules={[{ required: true, message: 'Nhập tên gốc' }]}
              >
                <Input
                  placeholder="Tên gốc"
                  value={entry.source}
                  onChange={(e) => updateEntry(index, 'source', e.target.value)}
                />
              </Form.Item>
              <Form.Item
                name={`translation_${index}`}
                label={index === 0 ? 'Tên thay thế' : ''}
                className="flex-1"
                rules={[{ required: true, message: 'Nhập tên thay thế' }]}
              >
                <Input
                  placeholder="Tên thay thế"
                  value={entry.translation}
                  onChange={(e) => updateEntry(index, 'translation', e.target.value)}
                />
              </Form.Item>
              {entries.length > 1 && (
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeEntry(index)}
                  className="mt-6"
                />
              )}
            </div>
          ))}
        </div>

        <Button type="dashed" onClick={addEntry} icon={<PlusOutlined />} className="w-full mt-4">
          Thêm Dòng
        </Button>
      </Form>
    </Modal>
  )
}

export default AddMappingNameModal

