import subprocess
import shutil
import os

def delete_files1():
    file_path1 = r"C:\keiba_AI\newget\all_2026_merged.csv"
    file_path2 = r"C:\keiba_AI\newget\renew.csv"
    dir_path =  r"C:\keiba_AI\newget\regetData"

    # ファイルが存在するか確認してから削除する
    if os.path.exists(file_path1):
        os.remove(file_path1)
        print(f"{file_path1} を削除しました。")
    else:
        print("ファイルが見つかりませんでした。")

    if os.path.exists(file_path2):
        os.remove(file_path2)
        print(f"{file_path2} を削除しました。")
    else:
        print("ファイルが見つかりませんでした。")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # ディレクトリごと中身を削除
        os.makedirs(dir_path)    # 必要であればディレクトリを再作成
        print("ディレクトリの中身を削除しました。")

def delete_files2():
    file_path1 = r"C:\keiba_AI\final\processed_12_data.csv"
    file_path2 = r"C:\keiba_AI\final\processed_data.csv"
    # ファイルが存在するか確認してから削除する
    if os.path.exists(file_path1):
        os.remove(file_path1)
        print(f"{file_path1} を削除しました。")
    else:
        print("ファイルが見つかりませんでした。")

    if os.path.exists(file_path2):
        os.remove(file_path2)
        print(f"{file_path2} を削除しました。")
    else:
        print("ファイルが見つかりませんでした。")



def run_task():
    # 実行したいディレクトリのパスを設定してください
    dir_path_a = r"C:\keiba_AI\newget"
    dir_path_b = r"C:\keiba_AI\final"
    
    try:
        delete_files1()#ファイルを削除
        delete_files2() #ファイルを削除

        # 1. ディレクトリAへ移動して.pyを実行
        print("--- ステップ1: gewget.pyを実行中 ---")
        os.chdir(dir_path_a)
        # subprocess.runでA.pyを実行し、完了まで待機
        subprocess.run(["python", "gewget.py"], check=True)
        print("gewget.pyの実行が完了しました。")

        print("--- ステップ2: regetCombine.pyを実行中 ---")
        os.chdir(dir_path_a)
        # subprocess.runでA.pyを実行し、完了まで待機
        subprocess.run(["python", "regetCombine.py"], check=True)
        print("regetCombine.pyの実行が完了しました。")

        print("--- ステップ3: create.pyを実行中 ---")
        os.chdir(dir_path_a)
        # subprocess.runでA.pyを実行し、完了まで待機
        subprocess.run(["python", "create.py"], check=True)
        print("create.pyの実行が完了しました。")
        
        print("--- ステップ4: createnewfulldata.pyを実行中 ---")
        os.chdir(dir_path_a)
        # subprocess.runでA.pyを実行し、完了まで待機
        subprocess.run(["python", "createnewfulldata.py"], check=True)
        print("createnewfulldata.pyの実行が完了しました。")

        # 2. ディレクトリBへ移動してB.pyを実行
        print("--- ステップ5: .pyを実行中 ---")
        os.chdir(dir_path_b)
        subprocess.run(["python", "awszawsz.py"], check=True)
        print("B.pyの実行が完了しました。")

    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")
    except FileNotFoundError as e:
        print(f"ファイルまたはディレクトリが見つかりません: {e}")

if __name__ == "__main__":
    run_task()