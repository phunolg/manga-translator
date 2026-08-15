import React, { useState, useEffect } from 'react'
import { Modal, Form, Input, Button, message, Select } from 'antd'
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { addressMatrixAPI } from '../services/api'
import { Character, AddressMatrix } from '../types'

interface AddAddressMatrixModalProps {
  visible: boolean
  storyName: string
  character: Character | null
  characters?: Character[]
  onCancel: () => void
  onSuccess: () => void
}

const AddAddressMatrixModal: React.FC<AddAddressMatrixModalProps> = ({
  visible,
  storyName,
  character,
  characters = [],
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [addressMatrix, setAddressMatrix] = useState<AddressMatrix>({})
  const [rowCount, setRowCount] = useState(1)
  
  // Lọc danh sách nhân vật, loại trừ nhân vật hiện tại (speaker)
  const availableCharacters = characters.filter(
    (char) => char.id !== character?.id
  )
  
  // Tạo options cho Select
  const characterOptions = availableCharacters.map((char) => ({
    label: char.source_name,
    value: char.source_name,
  }))
  
  // Thêm option "Những người còn lại" ở đầu danh sách
  characterOptions.unshift({
    label: 'Những người còn lại',
    value: '__OTHERS__',
  })
  
  // Thêm option "Khác" để có thể nhập tên tùy chỉnh
  characterOptions.push({
    label: 'Khác (nhập tên tùy chỉnh)',
    value: '__CUSTOM__',
  })

  const loadAddressMatrix = async () => {
    if (!character) return
    
    try {
      const response = await addressMatrixAPI.get(storyName, character.source_name)
      if (response.data?.address_matrix) {
        setAddressMatrix(response.data.address_matrix)
        // Convert to form format
        const formData: Record<string, string> = {}
        const entries = Object.entries(response.data.address_matrix)
        setRowCount(entries.length > 0 ? entries.length : 1)
        
        entries.forEach(([key, value], index) => {
          // Xử lý trường hợp "other" (Những người còn lại)
          if (key === 'other' || key === '__OTHERS__') {
            formData[`target_${index}`] = '__OTHERS__'
          }
          // Kiểm tra xem target có trong danh sách nhân vật không
          else if (availableCharacters.some((char) => char.source_name === key)) {
          formData[`target_${index}`] = key
          } else {
            // Nếu không có trong danh sách, sử dụng "__CUSTOM__" và điền vào custom_target
            formData[`target_${index}`] = '__CUSTOM__'
            formData[`custom_target_${index}`] = key
          }
          formData[`description_${index}`] = value
        })
        form.setFieldsValue(formData)
      } else {
        setRowCount(1)
      }
    } catch (error) {
      console.error('Error loading address matrix:', error)
    }
  }

  useEffect(() => {
    if (visible && character) {
      loadAddressMatrix()
    } else if (!visible) {
      // Reset form when modal closes
      form.resetFields()
      setAddressMatrix({})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, character])

  const handleSubmit = async () => {
    if (!character) return

    try {
      const values = await form.getFieldsValue()
      const matrix: AddressMatrix = {}
      
      // Convert form values to address matrix format
      Object.keys(values).forEach((key) => {
        if (key.startsWith('target_')) {
          const index = key.replace('target_', '')
          const target = values[key]
          const customTarget = values[`custom_target_${index}`]
          const description = values[`description_${index}`]
          
          // Xử lý các trường hợp đặc biệt
          let finalTarget: string
          if (target === '__CUSTOM__') {
            finalTarget = customTarget
          } else if (target === '__OTHERS__') {
            // Sử dụng "other" cho backend
            finalTarget = 'other'
          } else {
            finalTarget = target
          }
          
          if (finalTarget && description) {
            matrix[finalTarget] = description
          }
        }
      })

      setLoading(true)
      // Sử dụng merge thay vì replace để bổ sung vào dữ liệu hiện có
      await addressMatrixAPI.merge(storyName, character.source_name, matrix)
      form.resetFields()
      setAddressMatrix({})
      setRowCount(1)
      onSuccess()
    } catch (error: any) {
      message.error('Không thể cập nhật ma trận xưng hô')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const addRow = () => {
    setRowCount((prev) => prev + 1)
  }

  const removeRow = (index: number) => {
    form.setFieldsValue({
      [`target_${index}`]: undefined,
      [`description_${index}`]: undefined,
      [`custom_target_${index}`]: undefined,
    })
    // Giảm số lượng rows nếu cần
    const fields = getFormFields()
    if (fields.length > 1) {
      setRowCount((prev) => Math.max(1, prev - 1))
    }
  }

  const getFormFields = () => {
    const values = form.getFieldsValue()
    const fields: Array<{ index: number; target: string; description: string; isCustom: boolean }> = []
    const usedIndices = new Set<number>()
    
    // Lấy tất cả các index từ các field đã có trong form
    Object.keys(values).forEach((key) => {
      if (key.startsWith('target_')) {
        const index = parseInt(key.replace('target_', ''))
        if (!isNaN(index)) {
          usedIndices.add(index)
        }
      }
    })
    
    // Đảm bảo có đủ số lượng rows theo rowCount
    for (let i = 0; i < rowCount; i++) {
      usedIndices.add(i)
    }
    
    // Tạo fields từ các index đã có
    Array.from(usedIndices).forEach((index) => {
      const target = values[`target_${index}`] || ''
        const description = values[`description_${index}`] || ''
      const customTarget = values[`custom_target_${index}`] || ''
      const isCustom = target === '__CUSTOM__'
      const finalTarget = isCustom ? customTarget : target
      
      fields.push({ index, target: finalTarget, description, isCustom })
    })

    return fields.sort((a, b) => a.index - b.index)
  }
  
  const handleTargetChange = (index: number, value: string) => {
    form.setFieldsValue({
      [`target_${index}`]: value,
      [`custom_target_${index}`]: value === '__CUSTOM__' ? '' : undefined,
    })
  }

  if (!character) return null

  return (
    <Modal
      title={`Ma Trận Xưng Hô - ${character.source_name}`}
      open={visible}
      onOk={handleSubmit}
      onCancel={onCancel}
      confirmLoading={loading}
      okText="Lưu"
      cancelText="Hủy"
      width={700}
    >
      <Form form={form} layout="vertical" className="mt-4">
        <div className="space-y-4">
          {getFormFields().map((field) => {
            const targetValue = form.getFieldValue(`target_${field.index}`)
            const isCustom = targetValue === '__CUSTOM__'
            const isOthers = targetValue === '__OTHERS__'
            return (
            <div key={field.index} className="flex gap-2 items-start">
                <div className="flex-1">
              <Form.Item
                name={`target_${field.index}`}
                label={field.index === 0 ? 'Nhân Vật Đích' : ''}
                    rules={[{ required: true, message: 'Chọn nhân vật đích' }]}
                  >
                    <Select
                      placeholder="Chọn nhân vật đích"
                      options={characterOptions}
                      onChange={(value) => handleTargetChange(field.index, value)}
                      showSearch
                      filterOption={(input, option) =>
                        (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                      }
                    />
                  </Form.Item>
                  {isCustom && (
                    <Form.Item
                      name={`custom_target_${field.index}`}
                rules={[{ required: true, message: 'Nhập tên nhân vật đích' }]}
              >
                      <Input placeholder="Nhập tên nhân vật đích" />
              </Form.Item>
                  )}
                  {isOthers && (
                    <div className="text-sm text-gray-500 mt-1">
                      Áp dụng cho tất cả những người không được liệt kê cụ thể
                    </div>
                  )}
                </div>
              <Form.Item
                name={`description_${field.index}`}
                label={field.index === 0 ? 'Cách Xưng Hô' : ''}
                className="flex-1"
                rules={[{ required: true, message: 'Nhập cách xưng hô' }]}
              >
                <Input placeholder="Ví dụ: ta-cô, ta-công tử" />
              </Form.Item>
              {getFormFields().length > 1 && (
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeRow(field.index)}
                  className="mt-6"
                />
              )}
            </div>
            )
          })}
        </div>
        <Button
          type="dashed"
          onClick={addRow}
          icon={<PlusOutlined />}
          className="w-full mt-4"
        >
          Thêm Dòng
        </Button>
      </Form>
    </Modal>
  )
}

export default AddAddressMatrixModal

