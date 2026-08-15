import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { Table, Button, Select, Input, message, Space, Popconfirm, Tag } from 'antd'
import { DeleteOutlined, SaveOutlined } from '@ant-design/icons'
import { addressMatrixAPI } from '../services/api'
import { Character, AddressMatrix } from '../types'

interface AddressMatrixTableProps {
  storyName: string
  character: Character | null
  characters?: Character[]
  addressMatrix: AddressMatrix
  onUpdate: () => void
}

interface TableRow {
  key: string
  target: string
  address: string
  isNew?: boolean
  customTarget?: string
}

interface EditableCellProps {
  editing?: boolean
  dataIndex?: string
  title?: string
  record?: TableRow
  children: React.ReactNode
  inputType?: 'select' | 'text'
  editingValue?: string
  editingTargetValue?: string
  onEditingValueChange?: (value: string) => void
  characterOptions?: Array<{ label: string; value: string }>
  onTargetChange?: (value: string) => void
}

// Component EditableCell được tách riêng để tránh re-render
const EditableCell: React.FC<EditableCellProps> = React.memo(({ 
  editing = false, 
  dataIndex, 
  record, 
  children, 
  inputType,
  editingValue,
  editingTargetValue,
  onEditingValueChange,
  characterOptions = [],
  onTargetChange,
  ...restProps 
}) => {
  const [customTarget, setCustomTarget] = useState('')
  
  // Kiểm tra record có tồn tại không
  if (!record) {
    return <td {...restProps}>{children}</td>
  }
  
  // Sử dụng editingTargetValue từ props nếu có, nếu không thì dùng từ record
  const targetValue = editingTargetValue ?? record.target ?? ''

  if (editing) {
    if (dataIndex === 'target') {
      if (inputType === 'select') {
        return (
          <td {...restProps}>
            <div>
              <Select
                value={targetValue}
                onChange={(value) => {
                  onTargetChange?.(value)
                }}
                options={characterOptions}
                style={{ width: '100%' }}
                placeholder="Chọn nhân vật đích"
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
              />
              {targetValue === '__CUSTOM__' && (
                <Input
                  placeholder="Nhập tên nhân vật đích"
                  value={customTarget}
                  onChange={(e) => setCustomTarget(e.target.value)}
                  onBlur={() => {
                    if (customTarget.trim()) {
                      onTargetChange?.(customTarget.trim())
                    }
                  }}
                  style={{ marginTop: 8 }}
                  onPressEnter={(e) => {
                    const target = e.currentTarget.value.trim()
                    if (target) {
                      onTargetChange?.(target)
                    }
                  }}
                />
              )}
            </div>
          </td>
        )
      }
    } else if (dataIndex === 'address') {
      return (
        <td {...restProps}>
          <Input
            value={editingValue ?? ''}
            onChange={(e) => {
              onEditingValueChange?.(e.target.value)
            }}
            placeholder="Nhập cách xưng hô"
          />
        </td>
      )
    }
  }
  return <td {...restProps}>{children}</td>
}, (prevProps, nextProps) => {
  // Chỉ re-render khi các props quan trọng thay đổi
  if (prevProps.editing !== nextProps.editing) return false
  if (prevProps.record?.key !== nextProps.record?.key) return false
  if (prevProps.editingValue !== nextProps.editingValue) return false
  if (prevProps.editingTargetValue !== nextProps.editingTargetValue) return false
  if (prevProps.record?.address !== nextProps.record?.address) return false
  if (prevProps.record?.target !== nextProps.record?.target) return false
  return true
})

const AddressMatrixTable: React.FC<AddressMatrixTableProps> = ({
  storyName,
  character,
  characters = [],
  addressMatrix,
  onUpdate,
}) => {
  const [dataSource, setDataSource] = useState<TableRow[]>([])
  const [editingKey, setEditingKey] = useState<string>('')
  const [loading, setLoading] = useState(false)
  // State để lưu giá trị đang chỉnh sửa, tránh mất focus
  const [editingValues, setEditingValues] = useState<Record<string, { target?: string; address?: string }>>({})
  // Ref để lưu giá trị mới nhất, tránh closure issue
  const editingValuesRef = useRef<Record<string, { target?: string; address?: string }>>({})
  const dataSourceRef = useRef<TableRow[]>([])
  
  // Cập nhật ref mỗi khi state thay đổi
  useEffect(() => {
    editingValuesRef.current = editingValues
  }, [editingValues])
  
  useEffect(() => {
    dataSourceRef.current = dataSource
  }, [dataSource])

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

  useEffect(() => {
    // Convert address matrix to table rows
    const rows: TableRow[] = Object.entries(addressMatrix).map(([target, address]) => ({
      key: target,
      target: target === 'other' ? '__OTHERS__' : target,
      address,
      isNew: false,
    }))
    setDataSource(rows)
  }, [addressMatrix])

  const isEditing = (record: TableRow) => record.key === editingKey

  const edit = (record: TableRow) => {
    setEditingKey(record.key)
    // Khởi tạo giá trị editing từ record hiện tại
    setEditingValues({
      [record.key]: {
        target: record.target,
        address: record.address,
      }
    })
  }

  const cancel = () => {
    setEditingKey('')
    setEditingValues({})
  }

  const save = async (key: string) => {
    try {
      // Đọc giá trị mới nhất từ ref để tránh closure issue
      const currentEditingValue = editingValuesRef.current[key]
      const currentBaseRow = dataSourceRef.current.find((item) => item.key === key)
      
      if (!currentBaseRow) return
      
      // Ưu tiên giá trị từ editingValues nếu key tồn tại trong editingValues
      // Chỉ dùng giá trị từ editingValues nếu nó không phải empty string
      // Nếu editingValue tồn tại nhưng address/target là undefined/null/empty, vẫn dùng giá trị từ baseRow
      const targetValue = (currentEditingValue && currentEditingValue.target !== undefined && currentEditingValue.target !== null && currentEditingValue.target !== '') 
        ? currentEditingValue.target 
        : currentBaseRow.target
      const addressValue = (currentEditingValue && currentEditingValue.address !== undefined && currentEditingValue.address !== null && currentEditingValue.address !== '') 
        ? currentEditingValue.address 
        : currentBaseRow.address
      
      const row = {
        target: targetValue,
        address: addressValue,
      }

      // Debug log để kiểm tra giá trị
      console.log('Save - key:', key)
      console.log('Save - currentEditingValue:', currentEditingValue)
      console.log('Save - editingValuesRef.current:', editingValuesRef.current)
      console.log('Save - currentBaseRow:', currentBaseRow)
      console.log('Save - row:', row)
      console.log('Save - addressValue:', addressValue, 'type:', typeof addressValue, 'isString:', typeof addressValue === 'string')

      // Validate
      if (!row.target || row.target === '__CUSTOM__' || row.target === '') {
        message.error('Vui lòng chọn hoặc nhập nhân vật đích')
        return
      }
      
      // Kiểm tra address: phải là string và không rỗng sau khi trim
      const addressStr = String(row.address || '')
      if (!addressStr || addressStr.trim() === '') {
        console.error('Validation failed - address:', row.address, 'addressStr:', addressStr, 'trimmed:', addressStr.trim())
        message.error('Vui lòng nhập cách xưng hô')
        return
      }

      // Convert to address matrix format
      const targetKey = row.target === '__OTHERS__' ? 'other' : row.target
      const updateData: AddressMatrix = {
        [targetKey]: addressStr.trim(),
      }

      setLoading(true)
      await addressMatrixAPI.merge(storyName, character!.source_name, updateData)
      
      // Cập nhật dataSource với giá trị đã lưu
      setDataSource(prev => {
        const newData = [...prev]
        const index = newData.findIndex((item) => item.key === key)
        if (index > -1) {
          newData[index] = { ...newData[index], target: row.target, address: row.address }
        }
        return newData
      })
      
      setEditingKey('')
      setEditingValues({})
      message.success('Cập nhật thành công!')
      onUpdate()
    } catch (error: any) {
      message.error('Không thể cập nhật ma trận xưng hô')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (key: string) => {
    try {
      const row = dataSource.find((item) => item.key === key)
      if (!row) return
      
      // Xóa bằng cách replace toàn bộ trừ dòng bị xóa
      const newMatrix: AddressMatrix = {}
      dataSource.forEach((item) => {
        if (item.key !== key) {
          const itemTarget = item.target === '__OTHERS__' ? 'other' : item.target
          newMatrix[itemTarget] = item.address
        }
      })

      setLoading(true)
      await addressMatrixAPI.replace(storyName, character!.source_name, newMatrix)
      message.success('Xóa thành công!')
      onUpdate()
    } catch (error: any) {
      message.error('Không thể xóa ma trận xưng hô')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }



  const columns = [
    {
      title: 'Nhân Vật Đích',
      dataIndex: 'target',
      width: '40%',
      editable: true,
      render: (text: string) => {
        if (text === '__OTHERS__' || text === 'other') {
          return <Tag color="blue">Những người còn lại</Tag>
        }
        return text
      },
    },
    {
      title: 'Cách Xưng Hô',
      dataIndex: 'address',
      width: '40%',
      editable: true,
    },
    {
      title: 'Thao Tác',
      dataIndex: 'operation',
      render: (_: any, record: TableRow) => {
        const editable = isEditing(record)
        return editable ? (
          <Space>
            <Button
              type="link"
              onClick={() => save(record.key)}
              icon={<SaveOutlined />}
              loading={loading}
            >
              Lưu
            </Button>
            <Button type="link" onClick={cancel}>
              Hủy
            </Button>
          </Space>
        ) : (
          <Space>
            <Button type="link" onClick={() => edit(record)}>
              Chỉnh Sửa
            </Button>
            <Popconfirm
              title="Bạn có chắc chắn muốn xóa?"
              onConfirm={() => handleDelete(record.key)}
            >
              <Button type="link" danger icon={<DeleteOutlined />}>
                Xóa
              </Button>
            </Popconfirm>
          </Space>
        )
      },
    },
  ]

  // Callback để xử lý thay đổi giá trị address
  const handleAddressChange = useCallback((key: string, value: string) => {
    console.log('handleAddressChange - key:', key, 'value:', value)
    setEditingValues(prev => {
      const newValues = {
        ...prev,
        [key]: {
          ...(prev[key] || {}),
          address: value
        }
      }
      console.log('handleAddressChange - prev:', prev, 'newValues:', newValues)
      return newValues
    })
  }, [])

  // Callback để xử lý thay đổi target
  const handleTargetChange = useCallback((key: string, value: string) => {
    // Cập nhật editingValues để đảm bảo giá trị được lưu khi save
    setEditingValues(prev => ({
      ...prev,
      [key]: {
        ...prev[key],
        target: value
      }
    }))
    // Cập nhật dataSource để hiển thị ngay
    setDataSource(prev => {
      const newData = [...prev]
      const index = newData.findIndex((item) => item.key === key)
      if (index > -1) {
        newData[index].target = value
        if (value !== '__CUSTOM__') {
          newData[index].customTarget = undefined
        }
      }
      return newData
    })
  }, [])

  const mergedColumns = useMemo(() => columns.map((col) => {
    if (!col.editable) {
      return col
    }
    return {
      ...col,
      onCell: (record: TableRow | undefined) => {
        if (!record) {
          return {}
        }
        const editing = isEditing(record)
        const editingValue = editingValues[record.key]?.address
        const editingTargetValue = editingValues[record.key]?.target
        
        return {
          record,
          inputType: col.dataIndex === 'target' ? 'select' : 'text',
          dataIndex: col.dataIndex,
          title: col.title,
          editing,
          editingValue: col.dataIndex === 'address' ? editingValue : undefined,
          editingTargetValue: col.dataIndex === 'target' ? editingTargetValue : undefined,
          onEditingValueChange: col.dataIndex === 'address' 
            ? (value: string) => handleAddressChange(record.key, value)
            : undefined,
          characterOptions: col.dataIndex === 'target' ? characterOptions : undefined,
          onTargetChange: col.dataIndex === 'target'
            ? (value: string) => handleTargetChange(record.key, value)
            : undefined,
        }
      },
    }
  }), [editingValues, editingKey, characterOptions, handleAddressChange, handleTargetChange])

  if (!character) return null

  return (
    <div>
      <Table
        components={{
          body: {
            cell: EditableCell,
          },
        }}
        bordered
        dataSource={dataSource}
        columns={mergedColumns}
        rowClassName="editable-row"
        pagination={false}
        loading={loading}
      />
    </div>
  )
}

export default AddressMatrixTable

