import os
import time

import utils


if __name__ == "__main__":
    DB = utils.firebase()
    DB_BACKUP = utils.firebase(backup=True)

    # IMPORTANT: DO NOT DELETE THIS. MISCONFIGURATION CAN CAUSE `BACKUP` TO BE IDENTICAL
    # TO `DB` AND MAY RESULT IN ACCIDENTALLY WIPING OUT ALL THE PRODUCTION DATABASE.
    assert DB.project == "online-extremism"
    assert DB_BACKUP.project == "online-extremism-backup"

    for orig_collection_ref in DB.collections():
        collection_name = orig_collection_ref.id
        backup_collection_ref = DB_BACKUP.collection(collection_name)

        # 1: empty backup db collection.
        print(f"Emptying backup db collection `{collection_name}`...")
        for i, doc in enumerate(utils.timeout_stream(backup_collection_ref)):
            doc.reference.delete()
            if i % 100 == 0:
                print(i)

        # 2: fill backup by reading the contents of db.
        print(f"Copying db collection `{collection_name}` to backup...")
        for i, doc in enumerate(utils.timeout_stream(orig_collection_ref)):
            # Use `set` instead of `add` to keep the original ID's.
            backup_collection_ref.document(doc.id).set(doc.to_dict())
            if i % 100 == 0:
                print(i)
