# singletons/query_classifier.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from singletons.logger import get_logger

logger = get_logger()

_classifier_model = None

# Define classification categories with example phrases
PRODUCT_RELATED_EXAMPLES = [
	# Basic product inquiries (Vietnamese)
	"tôi muốn mua bánh",
	"có bánh gato không",
	"giá bánh bao nhiêu",
	"bánh ngọt nào ngon",
	"nước uống gì có",
	"menu đồ ăn",
	"sản phẩm mới",
	"đặt hàng bánh",
	"tìm bánh sinh nhật",
	"bánh mì baguette còn không?",
	"cho tôi xem các loại bánh kem",
	"bánh này có những vị gì?",
	"tôi muốn đặt một chiếc bánh cho ngày mai",
	"thông tin về bánh tiramisu",
	"bánh su kem giá sao?",
	"có loại bánh nào không đường không?",
	"tư vấn cho tôi một loại bánh ngon",
	"bộ sưu tập bánh mới nhất",
	
	# Extended Vietnamese product inquiries
	"bánh croissant có sẵn không?",
	"tôi cần đặt bánh cưới",
	"có bánh cho người ăn kiêng không?",
	"bánh macaron có mấy màu?",
	"tôi muốn xem bánh tart trái cây",
	"có đồ uống nóng không?",
	"bánh muffin chocolate còn không?",
	"giá một hộp cookie bao nhiêu?",
	"có bánh không gluten không?",
	"tôi muốn đặt 50 cái cupcake",
	"bánh éclair có nhân gì?",
	"nước ép trái cây tươi có không?",
	"có bánh vegan không?",
	"tôi cần bánh giao trong ngày",
	"bánh cheesecake có mấy loại?",
	"có cà phê và bánh combo không?",
	"bánh bông lan có size nào?",
	"tôi muốn custom bánh sinh nhật",
	"có bánh kẹo cho trẻ em không?",
	"smoothie có những vị gì?",
	
	# Basic product inquiries (English)
	"cookie chocolate",
	"pastry available",
	"dessert menu",
	"cake price",
	"food ordering",
	"what kind of bread do you have?",
	"show me your cupcakes",
	"what flavors does this cake come in?",
	"I'd like to order a cake for tomorrow",
	"details about tiramisu cake",
	"how much are cream puffs?",
	"any sugar-free cakes?",
	"recommend a delicious cake",
	"latest cake collection",
	
	# Extended English product inquiries
	"do you have fresh croissants?",
	"I need to order a wedding cake",
	"any diet-friendly options?",
	"what colors do macarons come in?",
	"I want to see fruit tarts",
	"do you serve hot beverages?",
	"are chocolate muffins available?",
	"how much for a box of cookies?",
	"any gluten-free options?",
	"I need to order 50 cupcakes",
	"what filling does éclair have?",
	"do you have fresh fruit juice?",
	"any vegan options available?",
	"I need same-day delivery",
	"how many types of cheesecake?",
	"coffee and pastry combo?",
	"what sizes for sponge cake?",
	"I want custom birthday cake",
	"kids' candy and treats?",
	"what smoothie flavors available?",
	
	# Specific product searches
	"red velvet cake price",
	"chocolate chip cookies bulk order",
	"espresso and croissant deal",
	"birthday cake with photo print",
	"lactose-free ice cream",
	"whole wheat bread available",
	"seasonal fruit tarts",
	"coffee bean types",
	"cake decoration options",
	"bulk order discount",
	"weekend special menu",
	"valentine's day collection",
	"christmas themed cakes",
	"corporate event catering",
	"afternoon tea set",
	
	# Product comparison and selection
	"compare chocolate vs vanilla",
	"which cake is most popular",
	"best seller recommendations",
	"difference between pastries",
	"most affordable options",
	"premium cake selection",
	"today's fresh items",
	"chef's special recommendation",
	"bestselling drinks",
	"signature desserts"
]

GENERAL_CONVERSATION_EXAMPLES = [
	# Basic greetings and polite conversation (Vietnamese)
	"xin chào",
	"cảm ơn",
	"bạn là ai",
	"bạn có khỏe không?",
	"hôm nay thời tiết thế nào?",
	"chúc một ngày tốt lành",
	"tạm biệt",
	"bạn có thể giúp tôi được không?",
	
	# Store logistics and general info (Vietnamese)
	"giờ mở cửa",
	"địa chỉ cửa hàng",
	"cách thanh toán",
	"chính sách đổi trả",
	"liên hệ",
	"cửa hàng có gần đây không?",
	"tôi có thể đậu xe ở đâu?",
	"có chương trình khuyến mãi gì không?",
	"cửa hàng có wifi không?",
	"tôi muốn nói chuyện với quản lý",
	
	# Extended Vietnamese general conversation
	"cửa hàng mở từ mấy giờ?",
	"có chỗ ngồi trong cửa hàng không?",
	"toilet ở đâu?",
	"có thể mang thú cưng vào không?",
	"bạn làm việc ở đây bao lâu rồi?",
	"cửa hàng có mấy chi nhánh?",
	"có thể đặt bàn trước không?",
	"nhạc nền hay quá",
	"không gian ở đây thật đẹp",
	"tôi đến lần đầu tiên",
	"có group chat khách hàng không?",
	"fanpage facebook của shop?",
	"có tuyển nhân viên không?",
	"làm thế nào để trở thành thành viên VIP?",
	"có app mobile không?",
	"website của cửa hàng là gì?",
	"có ship toàn quốc không?",
	"thời gian ship bao lâu?",
	"phí ship là bao nhiêu?",
	"có COD không?",
	
	# Basic greetings and polite conversation (English)
	"hello",
	"thank you",
	"how are you?",
	"what's the weather like today?",
	"have a nice day",
	"goodbye",
	"can you help me?",
	
	# Store logistics and general info (English)
	"store hours",
	"location",
	"payment methods",
	"return policy",
	"contact information",
	"is the store nearby?",
	"where can I park?",
	"any promotions running?",
	"do you have Wi-Fi?",
	"I'd like to speak to the manager",
	
	# Extended English general conversation
	"what time do you open?",
	"is there seating inside?",
	"where's the restroom?",
	"can I bring my pet inside?",
	"how long have you worked here?",
	"how many locations do you have?",
	"can I make a reservation?",
	"I love the background music",
	"this place looks beautiful",
	"this is my first time here",
	"do you have a customer group chat?",
	"what's your Facebook page?",
	"are you hiring?",
	"how to become a VIP member?",
	"do you have a mobile app?",
	"what's your website?",
	"do you deliver nationwide?",
	"how long does delivery take?",
	"what's the delivery fee?",
	"do you accept cash on delivery?",
	
	# Personal and casual conversation
	"I'm feeling tired today",
	"nice to meet you",
	"what's your name?",
	"how's your day going?",
	"I'm new to this area",
	"do you recommend this neighborhood?",
	"where's a good place to eat nearby?",
	"is there public transport here?",
	"what's the best time to visit?",
	"I love the interior design",
	"this place has great ambiance",
	"I'm here with my family",
	"celebrating my birthday today",
	"just browsing around",
	
	# Technology and service inquiries
	"how to use the QR code menu?",
	"is there phone charging station?",
	"can I get the receipt via email?",
	"do you have loyalty points system?",
	"what payment apps do you accept?",
	"can I split the bill?",
	"is there student discount?",
	"senior citizen discount available?",
	"military discount offered?",
	"group booking discounts?",
	
	# Feedback and reviews
	"I want to leave a review",
	"how can I provide feedback?",
	"where can I rate your service?",
	"I had a great experience",
	"staff here is very friendly",
	"I'll definitely come back",
	"I'll recommend this place",
	"can I post photos on social media?",
	"do you have Instagram account?",
	"what's your TikTok handle?"
]


def get_classifier_model():
    """Get the sentence transformer model for classification"""
    global _classifier_model
    if _classifier_model is None:
        try:
            # Use the same Vietnamese model you already have
            from singletons.embedding_model import get_embedding_model
            _classifier_model = get_embedding_model()
            logger.info(
                "Using existing embedding model for query classification")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise e
    return _classifier_model


class QueryClassifier:
    def __init__(self):
        self.model = get_classifier_model()
        self.product_embeddings = None
        self.general_embeddings = None
        self._initialize_category_embeddings()

    def _initialize_category_embeddings(self):
        """Pre-compute embeddings for category examples"""
        try:
            self.product_embeddings = self.model.encode(
                PRODUCT_RELATED_EXAMPLES,
                normalize_embeddings=True
            )
            self.general_embeddings = self.model.encode(
                GENERAL_CONVERSATION_EXAMPLES,
                normalize_embeddings=True
            )
            logger.info("Category embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize category embeddings: {e}")
            raise e

    def classify_query(self, query: str, threshold: float = 0.6):
        """
        Classify if a query is product-related or general conversation

        Args:
            query: User's input query
            threshold: Similarity threshold for classification

        Returns:
            dict: {
                'is_product_related': bool,
                'confidence': float,
                'category': str,
                'reasoning': str
            }
        """
        try:
            # Get query embedding
            query_embeddings = self.model.encode(
                [query], normalize_embeddings=True)
            query_embedding = query_embeddings[0]

            # Calculate similarities with both categories
            product_similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.product_embeddings
            )[0]

            general_similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.general_embeddings
            )[0]

            # Get max similarity for each category
            max_product_sim = np.max(product_similarities)
            max_general_sim = np.max(general_similarities)

            # Determine classification
            if max_product_sim > max_general_sim and max_product_sim > threshold:
                is_product_related = True
                confidence = float(max_product_sim)
                category = "product_inquiry"
                reasoning = f"Query shows high similarity ({confidence:.3f}) to product-related examples"
            elif max_general_sim > threshold:
                is_product_related = False
                confidence = float(max_general_sim)
                category = "general_conversation"
                reasoning = f"Query shows high similarity ({confidence:.3f}) to general conversation examples"
            else:
                # Default to product-related if unclear (conservative approach)
                is_product_related = True
                confidence = float(max(max_product_sim, max_general_sim))
                category = "unclear_defaulting_to_product"
                reasoning = f"Unclear classification, defaulting to product search (max sim: {confidence:.3f})"

            result = {
                'is_product_related': is_product_related,
                'confidence': confidence,
                'category': category,
                'reasoning': reasoning,
                'scores': {
                    'product_similarity': float(max_product_sim),
                    'general_similarity': float(max_general_sim)
                }
            }

            logger.info(f"Query classification: '{query[:50]}...' -> {result}")
            return result

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to product-related on error
            return {
                'is_product_related': True,
                'confidence': 0.0,
                'category': "error_defaulting_to_product",
                'reasoning': f"Classification error: {e}",
                'scores': {'product_similarity': 0.0, 'general_similarity': 0.0}
            }


# Global classifier instance
_query_classifier = None


def get_query_classifier():
    """Get or create the global query classifier instance"""
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier()
    return _query_classifier
