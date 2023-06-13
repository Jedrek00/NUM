# NUM

1. Instalacja pakietów (w środowisku wirtualnym): `pip install -r requirements.txt`. Jak nie chce ci się robić środowiska to potrzebujesz tensorflow, mlflow i tensorflow_datasets.

2. MLFlow: po uruchomieniu skryptu w konsoli wpisujesz `mlflow ui` i pod http://localhost:5000/ masz wyniki.
3. DVC: Aby pobrać najnowsze zmiany w datasecie wpisujesz w konsoli `dvc pull`. Wcześniej musisz mieć [zainstalowane](https://dvc.org/doc/install) dvc.
