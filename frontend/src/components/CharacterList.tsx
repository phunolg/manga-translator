import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, Empty, Button, Image } from 'antd'
import { UserOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { Character } from '../types'

interface CharacterListProps {
  characters: Character[]
  storyName: string
}

const CharacterList: React.FC<CharacterListProps> = ({
  characters,
  storyName,
}) => {
  const navigate = useNavigate()

  if (characters.length === 0) {
    return (
      <Empty
        description="Chưa có nhân vật nào"
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {characters.map((character) => (
        <Card
          key={character.id}
          hoverable
          className="shadow-sm cursor-pointer"
          onClick={() => navigate(`/story/${storyName}/character/${character.source_name}`)}
          cover={
            character.image_path ? (
              <Image
                src={character.image_path}
                alt={character.source_name}
                height={200}
                className="object-cover"
              />
            ) : (
              <div className="h-48 bg-gray-100 flex items-center justify-center">
                <UserOutlined className="text-6xl text-gray-400" />
              </div>
            )
          }
          actions={[
            <Button
              type="text"
              icon={<InfoCircleOutlined />}
              onClick={(e) => {
                e.stopPropagation()
                navigate(`/story/${storyName}/character/${character.source_name}`)
              }}
            >
              Chi Tiết
            </Button>,
      
          ]}
        >
          <Card.Meta
            title={
              <div className="text-lg font-semibold">{character.source_name}</div>
            }
            description={
              <div>
                {character.description && (
                  <p className="text-gray-600 mb-2 line-clamp-2">{character.description}</p>
                )}
              </div>
            }
          />
        </Card>
      ))}
    </div>
  )
}

export default CharacterList

