pipeline {
    agent any
    stages {
        stage('Install dependencies') {
            steps {
                echo 'Install dependencies on Jenkins server (maybe unnecessary if test runs inside Docker)'

		sh """
		pwd
		env
		source /var/lib/jenkins/py3env/bin/activate
		cd ${env.WORKSPACE}
		pip install -r requirements.txt

		echo ${env.JOB_NAME}
		mkdir -p /tmp/${env.JOB_NAME}
		cd /tmp/${env.JOB_NAME}
		rm -rf qcore
		git clone https://github.com/ucgmsim/qcore.git
		pip install --no-deps ./qcore/
		ln -s $HOME/data/testing/Empirical_Engine/sample0 ${env.WORKSPACE}

		"""
            }
        }
        stage('Run regression tests') {
            steps {
                echo 'Run pytest'
		sh """
		source /var/lib/jenkins/py3env/bin/activate
		cd ${env.WORKSPACE}/empirical
  		pytest --black --ignore=test --ignore=GMM_models
  		cd test
  		pytest -vs
		"""
            }
        }
    }

    post {
	always {
                echo 'Tear down the environments'
		sh """
		rm -rf /tmp/${env.JOB_NAME}/*
		docker container prune -f
		"""
            }
    }
}
