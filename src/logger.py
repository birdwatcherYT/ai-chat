import logging
import os


def get_logger(
    name: str, log_file: str | None = None, level=logging.INFO
) -> logging.Logger:
    """
    シンプルな設定のロガーを取得します。
    ファイル出力は任意で、log_fileをNoneにするとファイルには出力されません。

    Args:
        name (str): ロガーの名前。通常は呼び出し元のモジュール名 (__name__) を指定します。
        log_file (Optional[str]): ログ出力先のファイル名。
                                   Noneを指定するとファイル出力は行われません。デフォルトはNoneです。
        level (int): ログレベル。logging.DEBUG, logging.INFO, logging.WARNING,
                     logging.ERROR, logging.CRITICAL など。デフォルトは logging.INFO です。

    Returns:
        logging.Logger: 設定済みのロガーオブジェクト。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 親ロガーへの伝播を停止

    # 既存のハンドラーをクリア（ロガーを複数回取得する場合に重複を防ぐため）
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # フォーマッターの設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # コンソールハンドラーを追加 (常にコンソールには出力)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラーを条件付きで追加
    if log_file:
        # ログファイルのディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
