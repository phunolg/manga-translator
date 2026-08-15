import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, theme } from 'antd'
import {
  BookOutlined,
  TranslationOutlined,
} from '@ant-design/icons'
import type { MenuProps } from 'antd'
import StoryList from './pages/StoryList'
import StoryDetail from './pages/StoryDetail'
import CharacterDetail from './pages/CharacterDetail'
import TranslatePage from './pages/TranslatePage'
import EpisodeDetail from './pages/EpisodeDetail'
import './App.css'

const { Header, Content, Footer, Sider } = Layout

const siderStyle: React.CSSProperties = {
  overflow: 'auto',
  height: '100vh',
  position: 'sticky',
  insetInlineStart: 0,
  top: 0,
  bottom: 0,
  scrollbarWidth: 'thin',
  scrollbarGutter: 'stable',
}

const AppContent: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken()

  const [selectedKey, setSelectedKey] = useState(() => {
    if (location.pathname.startsWith('/story/')) {
      return '1'
    } else if (location.pathname === '/translate') {
      return '2'
    }
    return '1'
  })

  const items: MenuProps['items'] = [
    {
      key: '1',
      icon: React.createElement(BookOutlined),
      label: 'Quản lý Truyện',
    },
    {
      key: '2',
      icon: React.createElement(TranslationOutlined),
      label: 'Dịch Truyện',
    },
  ]

  const handleMenuClick: MenuProps['onClick'] = (e) => {
    const key = e.key
    setSelectedKey(key)
    
    if (key === '1') {
      navigate('/')
    } else if (key === '2') {
      navigate('/translate')
    }
  }

  return (
    <Layout hasSider>
      <Sider style={siderStyle}>
        <div className="demo-logo-vertical p-4 text-center text-white text-lg font-bold">
          Ứng dụng dịch truyện
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={items}
          onClick={handleMenuClick}
        />
      </Sider>
      <Layout>
        <Header style={{ padding: 0, background: colorBgContainer }} />
        <Content style={{ margin: '24px 16px 0', overflow: 'initial' }}>
          <div
            style={{
              padding: 24,
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
              minHeight: 'calc(100vh - 112px)',
            }}
          >
            <Routes>
              <Route path="/" element={<StoryList />} />
              <Route path="/story/:storyName" element={<StoryDetail />} />
              <Route path="/story/:storyName/character/:characterName" element={<CharacterDetail />} />
              <Route path="/story/:storyName/episode/:chapterNumber" element={<EpisodeDetail />} />
              <Route path="/translate" element={<TranslatePage />} />
            </Routes>
          </div>
        </Content>
        <Footer style={{ textAlign: 'center' }}>
          MagiV2 ©{new Date().getFullYear()} Metadata Dashboard
        </Footer>
      </Layout>
    </Layout>
  )
}

const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App

