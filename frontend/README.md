# MagiV2 Frontend - Metadata Dashboard

Dashboard quản lý metadata cho truyện tranh sử dụng React 18, TypeScript, Ant Design và Tailwind CSS.

## Cài Đặt

```bash
cd frontend
npm install
```

## Chạy Development Server

```bash
npm run dev
```

Ứng dụng sẽ chạy tại `http://localhost:3000`

## Build Production

```bash
npm run build
```

## Cấu Trúc Project

```
frontend/
├── src/
│   ├── components/          # Các component tái sử dụng
│   │   ├── AddCharacterModal.tsx
│   │   ├── AddAddressMatrixModal.tsx
│   │   ├── AddTranslateDictModal.tsx
│   │   ├── CharacterList.tsx
│   │   └── CreateStoryModal.tsx
│   ├── pages/               # Các trang chính
│   │   ├── StoryList.tsx
│   │   └── StoryDetail.tsx
│   ├── services/            # API services
│   │   └── api.ts
│   ├── types/               # TypeScript types
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

## Tính Năng

- ✅ Hiển thị danh sách truyện
- ✅ Xem chi tiết truyện và danh sách nhân vật
- ✅ Thêm nhân vật mới với ảnh
- ✅ Quản lý ma trận xưng hô (address matrix)
- ✅ Thêm từ điển dịch cho các ngôn ngữ mới

## API Endpoints

Backend API chạy tại `http://localhost:8008/api/v1`

- `POST /metadata/create-story` - Tạo truyện mới
- `POST /metadata/character` - Thêm nhân vật
- `GET /metadata/character/{story_name}/{character_name}/address-matrix` - Lấy ma trận xưng hô
- `POST /metadata/character/{story_name}/{character_name}/address-matrix` - Cập nhật ma trận xưng hô
- `PATCH /metadata/metadata/{story_name}` - Cập nhật metadata

## Tech Stack

- React 18
- TypeScript
- Ant Design 5
- Tailwind CSS 3
- Vite
- React Router DOM 6
- Axios

