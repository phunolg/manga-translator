import React, { useState, useEffect } from 'react'
import { Modal, Form, Select, Input, Button, Space, message, Alert } from 'antd'
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { storyAPI } from '../services/api'
import { TranslateDictItem } from '../types'

interface AddTranslateDictModalProps {
  visible: boolean
  storyName: string
  existingTranslateDicts?: TranslateDictItem[]
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

const AddTranslateDictModal: React.FC<AddTranslateDictModalProps> = ({
  visible,
  storyName,
  existingTranslateDicts = [],
  editingLanguage,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [dictEntries, setDictEntries] = useState<Array<{ key: string; value: string }>>([
    { key: '', value: '' },
  ])
  const [selectedLanguage, setSelectedLanguage] = useState<string | undefined>(editingLanguage)

  // Load existing dictionary when editing
  useEffect(() => {
    if (visible) {
      if (editingLanguage) {
        const existingDict = existingTranslateDicts.find(
          (dict) => dict.language === editingLanguage
        )
        if (existingDict) {
          const entries = Object.entries(existingDict.dictionary).map(([key, value]) => ({
            key,
            value,
          }))
          setDictEntries(entries.length > 0 ? entries : [{ key: '', value: '' }])
          form.setFieldsValue({ language: editingLanguage })
          setSelectedLanguage(editingLanguage)
        }
      } else {
        form.resetFields()
        setDictEntries([{ key: '', value: '' }])
        setSelectedLanguage(undefined)
      }
    }
  }, [visible, editingLanguage, existingTranslateDicts, form])

  // Check if language already has dictionary
  const existingLanguages = existingTranslateDicts.map((dict) => dict.language)
  const isEditing = !!editingLanguage
  const isLanguageExists = selectedLanguage && existingLanguages.includes(selectedLanguage) && !isEditing

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      const language = values.language
      
      // Convert entries to dict format
      const translateDict: Record<string, string> = {}
      dictEntries.forEach((entry) => {
        if (entry.key && entry.value) {
          translateDict[entry.key] = entry.value
        }
      })

      if (Object.keys(translateDict).length === 0) {
        message.error('Vui lòng thêm ít nhất một từ điển')
        return
      }

      setLoading(true)
      try {
        // Sử dụng POST endpoint mới để thêm từ điển
        await storyAPI.createTranslateDict(storyName, {
          language: language,
          dictionary: translateDict,
      })
      
        message.success(
          isEditing
            ? 'Cập nhật từ điển thành công!'
            : 'Thêm từ điển thành công!'
        )
      form.resetFields()
      setDictEntries([{ key: '', value: '' }])
        setSelectedLanguage(undefined)
      onSuccess()
      } catch (error: any) {
        throw error
      }
    } catch (error: any) {
      if (error.errorFields) {
        return
      }
      message.error('Không thể thêm từ điển')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const addEntry = () => {
    setDictEntries([...dictEntries, { key: '', value: '' }])
  }

  const removeEntry = (index: number) => {
    if (dictEntries.length > 1) {
      setDictEntries(dictEntries.filter((_, i) => i !== index))
    }
  }

  const updateEntry = (index: number, field: 'key' | 'value', value: string) => {
    const newEntries = [...dictEntries]
    newEntries[index][field] = value
    setDictEntries(newEntries)
    form.setFieldsValue({
      [`key_${index}`]: newEntries[index].key,
      [`value_${index}`]: newEntries[index].value,
    })
  }

  const handleLanguageChange = (value: string) => {
    setSelectedLanguage(value)
    // If switching to a language that already has dictionary, load it
    const existingDict = existingTranslateDicts.find((dict) => dict.language === value)
    if (existingDict && !isEditing) {
      const entries = Object.entries(existingDict.dictionary).map(([key, value]) => ({
        key,
        value,
      }))
      setDictEntries(entries.length > 0 ? entries : [{ key: '', value: '' }])
    } else if (!existingDict) {
      setDictEntries([{ key: '', value: '' }])
    }
  }

  return (
    <Modal
      title={isEditing ? 'Chỉnh Sửa Từ Điển Dịch' : 'Thêm Từ Điển Dịch'}
      open={visible}
      onOk={handleSubmit}
      onCancel={() => {
        form.resetFields()
        setDictEntries([{ key: '', value: '' }])
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
            message="Ngôn ngữ này đã có từ điển"
            description="Các từ mới sẽ được thêm vào từ điển hiện có. Từ trùng lặp sẽ được cập nhật."
            type="info"
            showIcon
            className="mb-4"
          />
        )}

        <div className="space-y-4">
          <div className="font-medium mb-2">Từ Điển:</div>
          {dictEntries.map((entry, index) => (
            <div key={index} className="flex gap-2 items-start">
              <Form.Item
                name={`key_${index}`}
                label={index === 0 ? 'Từ Gốc' : ''}
                className="flex-1"
                rules={[{ required: true, message: 'Nhập từ gốc' }]}
              >
                <Input
                  placeholder="Từ gốc"
                  value={entry.key}
                  onChange={(e) => updateEntry(index, 'key', e.target.value)}
                />
              </Form.Item>
              <Form.Item
                name={`value_${index}`}
                label={index === 0 ? 'Nghĩa Dịch' : ''}
                className="flex-1"
                rules={[{ required: true, message: 'Nhập nghĩa dịch' }]}
              >
                <Input
                  placeholder="Nghĩa dịch"
                  value={entry.value}
                  onChange={(e) => updateEntry(index, 'value', e.target.value)}
                />
              </Form.Item>
              {dictEntries.length > 1 && (
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

        <Button
          type="dashed"
          onClick={addEntry}
          icon={<PlusOutlined />}
          className="w-full mt-4"
        >
          Thêm Từ
        </Button>
      </Form>
    </Modal>
  )
}

export default AddTranslateDictModal

