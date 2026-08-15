## Tổng quan API

Backend phục vụ dưới tiền tố `/api/v1`. Các phần dưới mô tả từng endpoint gồm: mục đích, cấu trúc input (JSON hoặc form-data) và cấu trúc output `ApiResponse`.

> Ghi chú: tất cả controller sử dụng `Depends(get_db)` để đảm bảo phiên làm việc với database đồng bộ/async an toàn.

---

### Metadata Controller (`/api/v1/metadata`)

1. **GET /metadata/stories**
   - **Input Schema**: không yêu cầu query/body.
   - **Output Schema**:
```json
{
  "status_code": 200,
  "is_successful": true,
  "data": [
    {
      "story_name": "Tên truyện",
      "story_type": "BASIC | ROMANCE | ...",
      "source_language": "VIETNAMESE | ENGLISH | ..."
    }
  ]
}
```
   - **Chức năng**: lấy danh sách truyện đang có.

2. **GET /metadata/stories/{story_name}**
   - **Input Schema**:
```json
{
  "path": {
    "story_name": "Tên truyện"
  }
}
```
   - **Output Schema** (rút gọn các trường chính):
```json
{
  "status_code": 200,
  "is_successful": true,
  "data": {
    "story_name": "...",
    "story_type": "...",
    "source_language": "...",
    "characters": [
      {
        "id": 1,
        "source_name": "Dư Lạc",
        "description": "...",
        "image_path": "/api/v1/files/images/...",
        "created_at": "2024-05-01T00:00:00",
        "updated_at": "2024-05-02T00:00:00"
      }
    ],
    "episodes": [
      {
        "id": 10,
        "chapter_number": 1,
        "created_at": "...",
        "updated_at": "..."
      }
    ],
    "translate_dicts": [
      {
        "language": "VIETNAMESE",
        "dictionary": {
          "hello": "xin chào"
        }
      }
    ],
    "mapping_names": [
      {
        "language": "ENGLISH",
        "dictionary": {
          "Dư Lạc": "Du Lac"
        }
      }
    ]
  }
}
```
   - **Chức năng**: cung cấp đầy đủ metadata để render trang StoryDetail.

3. **POST /metadata/stories**
   - **Input Schema**:
```json
{
  "story_name": "Tên truyện",
  "story_type": "BASIC | ROMANCE | ...",
  "source_language": "VIETNAMESE | ENGLISH | ..."
}
```
   - **Output Schema**: giống cấu trúc `GET /metadata/stories` nhưng chỉ chứa record vừa tạo.
   - **Chức năng**: khởi tạo truyện mới.

4. **POST /metadata/stories/{story_name}/translate-dict**
   - **Input Schema**:
```json
{
  "path": {
    "story_name": "Tên truyện"
  },
  "body": {
    "language": "VIETNAMESE",
    "dictionary": {
      "speaker": "dịch"
    }
  }
}
```
   - **Output Schema**: `ApiResponse` chứa `{ "language": ..., "dictionary": {...} }` sau merge.
   - **Chức năng**: thêm/ghi đè từ điển dịch theo ngôn ngữ.

5. **PATCH /metadata/{story_name}**
   - **Input Schema** (form-data, tất cả đều tùy chọn):
```json
{
  "path": {
    "story_name": "Tên truyện"
  },
  "form": {
    "mapping_name": "{\"ENGLISH\":{\"Dư Lạc\":\"Du Lac\"}}",
    "translate_dict": "{\"language\":{...}}",
    "story_type": "BASIC",
    "source_language": "VIETNAMESE"
  }
}
```
   - **Output Schema**: `ApiResponse` báo thành công/thất bại.
   - **Chức năng**: cập nhật metadata nâng cao.

6. **GET /metadata/stories/{story_name}/mapping-names**
   - **Input Schema**: path `story_name`, query optional `language`.
   - **Output Schema**:
```json
{
  "status_code": 200,
  "is_successful": true,
  "data": [
    {
      "language": "vietnamese",
      "dictionary": {
        "Thái Thư Lệ": "Thư Lệ",
        "Quách Thiên Khải": "Thiên Khải"
      }
    }
  ]
}
```
   - **Chức năng**: liệt kê toàn bộ mapping name (tên riêng → tên dịch) của một truyện, phục vụ UI quản lý tương tự từ điển dịch.

7. **POST /metadata/stories/{story_name}/mapping-names**
   - **Input Schema**:
```json
{
  "path": { "story_name": "..." },
  "body": {
    "language": "english",
    "dictionary": {
      "Dư Lạc": "Du Lac",
      "Thái Thư Lệ": "Tai Thu Le"
    }
  }
}
```
   - **Output Schema**: `ApiResponse` chứa `{ "language": "...", "dictionary": { ... } }` sau khi merge/ghi đè các entry trùng key.
   - **Chức năng**: thêm mới hoặc cập nhật hàng loạt mapping name cho một ngôn ngữ.

8. **DELETE /metadata/stories/{story_name}/mapping-names**
   - **Input Schema**: query `language`, `source` (ví dụ `?language=english&source=D%C6%B0%20L%E1%BA%A1c`).
   - **Output Schema**: `ApiResponse` message thành công hoặc lỗi nếu không tồn tại.
   - **Chức năng**: xoá một entry cụ thể khỏi bảng mapping name.

---

### Character Controller (`/api/v1/character`)

1. **POST /character**
   - **Input Schema** (multipart form-data): `story_name`, `name_character`, `description?`, `character_image` (file ảnh).
   - **Output Schema**:
```json
{
  "status_code": 201,
  "is_successful": true,
  "data": {
    "id": 12,
    "source_name": "Dư Lạc",
    "description": "...",
    "image_path": "/api/v1/files/images/...",
    "created_at": "...",
    "updated_at": "..."
  }
}
```
   - **Chức năng**: tạo nhân vật mới cho truyện.

2. **GET /character/{story_name}/{character_name}**
   - **Input Schema**: path `story_name`, `character_name`.
   - **Output Schema**: `ApiResponse` chứa `{ "character": {...}, "address_matrix": {"target": "description"} }`.
   - **Chức năng**: lấy chi tiết nhân vật + address matrix (cho trang quản lý).

3. **PATCH /character/{story_name}/{character_name}**
   - **Input Schema**: path params; form-data tùy chọn `name`, `description`, `character_image` (file mới).
   - **Output Schema**: giống `POST /character` nhưng dữ liệu sau cập nhật.
   - **Chức năng**: chỉnh sửa thông tin nhân vật hiện có.

4. **GET /character/{story_name}/{character_name}/address-matrix**
   - **Input Schema**: path params.
   - **Output Schema**: `ApiResponse` với field `address_matrix` dạng dictionary.
   - **Chức năng**: phục vụ bảng AddressMatrixTable (hiển thị dữ liệu gốc).

5. **POST /character/{story_name}/{character_name}/address-matrix**
   - **Input Schema**:
```json
{
  "path": {
    "story_name": "...",
    "character_name": "..."
  },
  "body": {
    "Dư Lạc": "ta-ngươi",
    "other": "ta-cô"
  }
}
```
   - **Output Schema**: `ApiResponse` báo thành công, kèm dữ liệu mới nếu service trả về.
   - **Chức năng**: merge/bổ sung entry address matrix.

6. **PUT /character/{story_name}/{character_name}/address-matrix**
   - **Input Schema**: giống POST nhưng replace toàn bộ.
   - **Chức năng**: ghi đè address matrix hiện tại.

7. **DELETE /character/{story_name}/{character_name}/address-matrix/{target_name}**
   - **Input Schema**: path params.
   - **Output Schema**: `ApiResponse` với message thành công hoặc lỗi.
   - **Chức năng**: xóa một entry xưng hô cụ thể.

---

### Episode Controller (`/api/v1/episode`)

1. **POST /episode**
   - **Input Schema**:
     - Multipart form: `chapter_pages` (danh sách UploadFile), `story_name` (string), `chapter_number` (int).
   - **Output Schema**:
```json
{
  "status_code": 201,
  "is_successful": true,
  "data": {
    "episode_id": 5,
    "chapter_number": 1,
    "pages_count": 12
  }
}
```
   - **Chức năng**: nhận ảnh chapter → chạy `get_transcript` → lưu ảnh vào `data/stories/<story>/chapters/<chapter>/...` và ghi metadata (page, transcript, bbox) vào DB.

2. **GET /episode/{story_name}/chapters/{chapter_number}/translated**
   - **Input Schema**:
```json
{
  "path": {
    "story_name": "...",
    "chapter_number": 10
  },
  "query": {
    "language": "VIETNAMESE | ENGLISH | ...",
    "pages": [1, 5, 7],
    "mode": "zip | inline",
    "translate": true
  }
}
```
     - `language` mặc định `VIETNAMESE`.
     - `pages` là danh sách `page_number` trong DB; truyền nhiều giá trị bằng `?pages=1&pages=5`. Nếu bỏ trống → dịch toàn bộ chapter.
     - `mode` mặc định `zip`. Nếu chọn `inline`, API trả JSON chứa base64 ảnh để frontend hiển thị trực tiếp (ảnh gốc bên trái, ảnh đã dịch bên phải từng hàng).
     - `translate` (bool) mặc định `false`. Khi `false`, API tái sử dụng bản dịch đã lưu trong DB; khi `true`, sẽ dịch lại các trang được chọn và ghi đè dữ liệu cũ.
   - **Output Schema**:
     - `mode=zip`: trả về tệp `zip` chứa ảnh đã dịch.
     - `mode=inline`: 
```json
{
  "mode": "inline",
  "pages": [
    {
      "page_name": "page_1.png",
      "original": "data:image/png;base64,...",
      "translated": "data:image/png;base64,..."
    }
  ]
}
```
   - **Chức năng**: dịch và typeset tập truyện; có thể giới hạn chỉ các trang mong muốn và lựa chọn kiểu trả về (tải tệp hoặc hiển thị trực tiếp trên UI).

---

### File Controller (`/api/v1/files`)

1. **GET /files/images/{file_path}**
   - **Input Schema**:
```json
{
  "path": {
    "file_path": "D%C6%B0%20L%E1%BA%A1c%20v%C3%A0%20nh%E1%BB%AFng%20ng%C6%B0%E1%BB%9Di%20b%E1%BA%A1n/Character/d%C6%B0%20l%E1%BA%A1c.png"
  }
}
```
   - **Output Schema**: trả về trực tiếp file ảnh (HTTP file response, Content-Type theo extension). Nếu không hợp lệ => 403/404.
   - **Chức năng**: cung cấp ảnh tĩnh (nhân vật, chapter, ảnh dịch, ảnh gốc) cho frontend.

---

