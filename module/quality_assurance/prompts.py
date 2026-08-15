import json

from module.quality_assurance.type import ErrorType

check_quality_by_llm_prompt_user = """
{sourceLang} source: {source}
{targetLang} translation: {translation}
Image: [ảnh kèm theo từ truyện]
Hãy xem xét cả text và hình ảnh để đánh giá.
"""

# Mô tả chi tiết cho từng loại lỗi
error_descriptions = {
    "system_prompt": """
    # Bạn là một người đánh giá (annotator) chất lượng bản dịch máy trong truyện tranh (manga/manhua/manhwa).
    # Bạn sẽ nhận được:
    - Source segment (câu gốc)
    - Machine translation (bản dịch máy)
    - Context (các đoạn thoại trước và sau)
    - Inside (ý muốn, ý định, suy nghĩ của mỗi lời nói)
        
    # Nhiệm vụ:
    - Bạn CHỈ tập trung vào việc kiểm tra lỗi dựa trên yêu cầu của người dùng trong bản dịch. Không kiểm tra các loại lỗi khác.
    - Mọi đánh giá và đề xuất chỉnh sửa **phải bám sát nội dung gốc, không được lược bỏ hoặc làm mất ý**.
    - Nếu một segment không có lỗi nào thì bỏ qua, không đưa vào kết quả.
    
    # Đầu ra:
    Trả về một JSON array, mỗi phần tử ứng với một segment có lỗi.
    Mỗi phần tử có format như sau:
    Ví dụ:
    ```json
    [{
    "original_segment": "<câu gốc>",
    "translation_segment": "<bản dịch>",
    "errors": [
        {
            "error_type": "<nhóm lỗi>",
            "subcategory": "<phân loại con>",
            "description": "<giải thích ngắn>",
            "suggestion": "<cách sửa gọn gàng>"
        }
    ]
    }]
    ```
    # Output rules:
    - Luôn trả JSON array hợp lệ.
    - Không giải thích ngoài JSON.
    - Nếu không phát hiện lỗi => trả [].
    - Không bịa ra loại lỗi khác, chỉ dùng đúng nhóm đã cung cấp.

    # Quy tắc quan trọng:
    - Chỉ xuất hiện errors nếu có ít nhất 1 lỗi.
    - Nếu một segment không có lỗi => KHÔNG xuất hiện trong kết quả.
    - Nếu toàn bộ các segment không có lỗi => kết quả cuối cùng phải là một mảng rỗng [].
    - Chỉ đánh giá trên câu dịch, không được đánh giá trên câu gốc.
    - Yêu cầu chỉ đánh giá, phân loại những lỗi dựa trên nhóm được cung cấp. Nếu có lỗi khác thì không được đánh giá.
    """,
    ErrorType.ACCURACY: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Context: {context}
    Inside: {inside}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":

    - thêm nghĩa  
    Vd:  
    Source: "She is waiting at the door."  
    Translation: "Cô ấy đang đợi ở cửa với một người bạn."  
    => Lỗi: thêm chi tiết "với một người bạn" không có trong câu gốc.  

    - dịch sai  
    Vd:  
    Source: "He has already gone outside."  
    Translation: "Anh ấy chưa đi ra ngoài."  
    => Lỗi: dịch trái nghĩa, 'already gone' => 'chưa đi'.  

    - bỏ sót  
    Vd:  
    Source: "She bought apples and oranges."  
    Translation: "Cô ấy mua táo."  
    => Lỗi: bỏ sót 'oranges'.  

    - dùng sai đối tượng  
    Vd:  
    Source: "The wind is whispering."  
    Translation: "Người đàn ông đang thì thầm."  
    => Lỗi: từ mô tả cảnh vật (gió) bị dịch thành nhân vật (người đàn ông).  

    - dịch sai câu hỏi thành câu khẳng định/cảm thán  
    Vd:  
    Source: "Did you see him?"  
    Translation: "Tôi đã thấy anh ta rồi!"  
    => Lỗi: câu hỏi bị dịch thành câu cảm thán/khẳng định.  

    - dịch sai nghĩa bóng, ẩn dụ, văn phong tắt  
    Vd:  
    Source: "He finally kicked the bucket." (idiom: chết)  
    Translation: "Cuối cùng anh ta đã đá cái xô."  
    => Lỗi: dịch nghĩa đen, bỏ qua nghĩa bóng.  

    # Chú ý: 
    - TUYỆT ĐỐI không được xét lỗi khác tên riêng, không được xét lỗi xưng hô.
    - Không được xét lỗi liên quan đến tên tổ chức, thuật ngữ, hoặc tên nhân vật (những lỗi này đã thuộc nhóm NAMING_CONVENTION hoặc MISSING_TRANSLATION).
    """
    ,
    ErrorType.FLUENCY: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":

    - ngữ pháp không đúng với {targetLang}  
    Vd:  
    Source: "He goes to school every day."  
    Translation: "Anh ấy đi học ngày mỗi."  
    => Lỗi: sai trật tự từ trong tiếng Việt, đáng lẽ phải là "Anh ấy đi học mỗi ngày."  
 
    - lặp từ quá nhiều  
    Vd 1 (LỖI):   
    Translation: "Người này có liên quan đến công việc này đúng không"  
    => Lỗi: lặp từ "này" quá nhiều, gây rườm rà không tự nhiên. Dùng từ khác thay thế từ "này".
    Vd 2 (Đúng):  
    Translation: "Một tháng trước, các môn đệ của chúng ta đi mua đồ đạc và tài nguyên."  
    => Đúng: từ "mua" và "tài nguyên" không phải lặp, câu hoàn toàn tự nhiên, KHÔNG coi là lỗi.  

    - đọc bị gượng gạo (thiếu tự nhiên, dịch word-by-word)  
    Vd:  
    Source: "She burst into tears."  
    Translation: "Cô ấy nổ tung thành những giọt nước mắt."  
    => Lỗi: dịch word-by-word gây gượng gạo, tự nhiên hơn phải là "Cô ấy òa khóc."   
    """,
    ErrorType.CONTEXT: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Inside: {inside}
    Context: {context}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":
    - Câu dịch không liên quan đến hành động, bối cảnh hiện tại/trước/sau

    Ví dụ:
    1. Source: "Miss, you're mistaken."
    Translation: "Cô ấy cầm cốc rượu rồi ném xuống sàn."
    Inside: Mô tả cảnh Dư Lạc đang nói chuyện nghiêm túc với Dư Y Ba.
    Context: Các trang trước cho thấy hai nhân vật đang tranh luận về danh tính.
    => Lỗi: Câu dịch "ném cốc rượu" không liên quan đến source và cũng không khớp với ngữ cảnh (ở đây không có chi tiết nào về rượu hay hành động bạo lực).

    3. Source: "This person... his something to do with the task here?"
    Translation: "Cô ta hỏi anh ta có muốn uống rượu không."
    Inside: Tuyết Ly đang nghi ngờ vai trò của Dư Lạc trong nhiệm vụ.
    Context: Các nhân vật đang bàn về nhiệm vụ liên quan đến Yuanmo SEC.
    => Lỗi: Câu dịch thêm nội dung "uống rượu" không có trong source, đồng thời **lạc ngữ cảnh** vì không liên quan đến nhiệm vụ.

    4. Source: "You're finally here sir! I've waited so long!"
    Translation: "Cuối cùng anh cũng đến rồi! Em đã đợi lâu lắm!"
    Inside: Dư Y Ba tỏ ra vui mừng khi gặp nhân vật khác.
    Context: Ngay sau đó, nhân vật kia lại phủ nhận "I DON'T KNOW YOU!"
    => Không lỗi: Câu dịch khớp với cảm xúc, và **liên quan đến mạch truyện** khi dẫn đến sự đối lập ở lời thoại sau.
    """,
    
    ErrorType.NAMING_CONVENTION: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Context: {context}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":
    - Sử dụng sai đại từ chỉ giới tính
    Vd: 
    Source: "You are a good boy."  
    Translation: "Anh là một người tốt."  
    => Lỗi: Nhân vật đang nghe là giới tính nữ, nhưng dùng đại từ "anh" là sai, phải đại từ dành cho nữ.

    - Sai số lượng người nói (Ví dụ sử dụng đại từ số ít cho nhiều người hoặc ngược lại)
    """,
    ErrorType.STYLE: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Thể loại truyện: {genre}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":
    - Dùng từ không phù hợp với thể loại truyện
    - thuật ngữ không phù hợp ngữ cảnh
    Ví dụ:
    1. Source: "<Dư Lạc>: Miss, you're mistaken."
    Translation: "Cô công tử, cô hiểu lầm rồi."
    Genre: truyện cổ trang
    => Lỗi: "công tử" dùng cho nam nhưng lại gán cho nữ không phù hợp với cách xưng hô trong truyện cổ trang. Câu gốc chỉ cần "tiểu thư" hoặc "cô nương".
    
    2. Thể loại hiện đại
    Source: "Brother Zhang, please help me."
    Translation: "Huynh Trương, xin giúp ta."
    => Sai: "Huynh" là từ cổ trang, không hợp với truyện hiện đại. Nên dịch thành "Anh Trương" hoặc "Anh Zhang".

    Source: "We need to complete this mission quickly."
    Translation: "Chúng ta phải hoàn thành sứ mệnh này thật nhanh."
    => Sai: "sứ mệnh" nghe quá trang trọng, phong cách thiên sử thi. Trong truyện hiện đại, nên dùng "nhiệm vụ".
    """,
    
    ErrorType.MISSING_TRANSLATION: """
    Câu gốc: {source}
    Câu dịch: {translation}
    Từ điển tên riêng: {name_dictionary}
    Từ điển thuật ngữ: {term_dictionary}
    Hãy dựa vào câu dịch và câu gốc xác định các lỗi thuộc nhóm "{error_type}":
    - Dựa vào bộ từ điển được cung cấp gồm tên nhân vật, vật phẩm, tên riêng, thuật ngữ, ... để xác định các lỗi dịch sai. 
    - CHÚ Ý: Nếu tên riêng hoặc thuật ngữ đã được dịch đúng theo thì coi là HOÀN TOÀN CHÍNH XÁC, KHÔNG đưa vào bất kỳ nhóm lỗi nào khác.

    # Ví dụ:
    - Thuật ngữ chưa được dịch
    Source: "He obtained the Fire Lotus Core."
    Từ điển thuật ngữ: {{ "Fire Lotus Core": "Hỏa Liên Tâm" }}
    Translation: "Anh ta nhận được Fire Lotus Core."
    => Sai: "Fire Lotus Core" để nguyên, chưa dịch theo từ điển, phải là "Hỏa Liên Tâm".

    - Thuật ngữ dịch sai
    Source: "He obtained the Fire Lotus Core."
    Từ điển thuật ngữ: {{ "Fire Lotus Core": "Hỏa Liên Tâm" }}
    Translation: "Anh ta nhận được Hỏa Diêm Thuật."
    => Sai: dịch "Fire Lotus Core" thành "Hỏa Diêm Thuật" là sai, phải là "Hỏa Liên Tâm".

    - Thuật ngữ dịch ĐÚNG (KHÔNG PHẢI LỖI)
    Source: "He obtained the Fire Lotus Core."
    Từ điển thuật ngữ: {{ "Fire Lotus Core": "Hỏa Liên Tâm" }}
    Translation: "Anh ta nhận được Hỏa Liên Tâm."
    => Đúng: đã dịch chính xác theo từ điển, KHÔNG được gán vào bất kỳ nhóm lỗi nào.
    """

}

