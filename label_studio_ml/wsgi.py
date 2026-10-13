
import os

from label_studio_ml.api import init_app
from model import PhoBERTNERModel

app = init_app(model_class=PhoBERTNERModel)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "9090")),
        debug=False,
    )
