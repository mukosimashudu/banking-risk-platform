from sqlalchemy import text
from src.config.db import engine


def create_clean_transactions():
    print("Creating clean transactions...")

    query = """
    DROP TABLE IF EXISTS staging.clean_ieee_transactions;

    SELECT
        TRY_CAST(TransactionID AS BIGINT) AS transaction_id,
        TRY_CAST(isFraud AS INT) AS is_fraud,
        TRY_CAST(TransactionDT AS FLOAT) AS transaction_dt,
        TRY_CAST(TransactionAmt AS FLOAT) AS transaction_amt,
        ProductCD,
        TRY_CAST(card1 AS FLOAT) AS card1,
        TRY_CAST(card2 AS FLOAT) AS card2,
        TRY_CAST(card3 AS FLOAT) AS card3,
        card4,
        TRY_CAST(card5 AS FLOAT) AS card5,
        card6,
        TRY_CAST(addr1 AS FLOAT) AS addr1,
        TRY_CAST(addr2 AS FLOAT) AS addr2,
        P_emaildomain,
        R_emaildomain
    INTO staging.clean_ieee_transactions
    FROM raw.ieee_train_transaction;
    """

    with engine.begin() as conn:
        conn.execute(text(query))

    print("Clean transactions created.")


def create_clean_identity():
    print("Creating clean identity...")

    query = """
    DROP TABLE IF EXISTS staging.clean_ieee_identity;

    SELECT
        TRY_CAST(TransactionID AS BIGINT) AS transaction_id,
        DeviceType,
        DeviceInfo,
        id_31
    INTO staging.clean_ieee_identity
    FROM raw.ieee_train_identity;
    """

    with engine.begin() as conn:
        conn.execute(text(query))

    print("Clean identity created.")


def create_final_dataset():
    print("Creating final fraud dataset...")

    query = """
    DROP TABLE IF EXISTS staging.ieee_fraud_final;

    SELECT
        t.*,
        i.DeviceType,
        i.DeviceInfo,
        i.id_31
    INTO staging.ieee_fraud_final
    FROM staging.clean_ieee_transactions t
    LEFT JOIN staging.clean_ieee_identity i
        ON t.transaction_id = i.transaction_id;
    """

    with engine.begin() as conn:
        conn.execute(text(query))

    print("Final fraud dataset created.")


def main():
    create_clean_transactions()
    create_clean_identity()
    create_final_dataset()
    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()