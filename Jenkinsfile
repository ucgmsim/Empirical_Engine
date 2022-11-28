pipeline {
    agent any

    environment {
        TEMP_DIR="/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}"
    }
    stages {
        stage('Setting up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    echo "[ Current directory ] : " `pwd`
                    echo "[ Environment Variables ] "
                    env
# Each stage needs custom setting done again. By default /bin/python is used.
                    source /home/qcadmin/py310/bin/activate
                    mkdir -p $TEMP_DIR
                    python -m venv $TEMP_DIR/venv
# activate new virtual env
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Install dependencies ]"
# This can cause storage going overflow. OpenQuake needs lots of temp storage
                    pip install -r requirements.txt
                    echo "[ Install qcore ]"
                    cd $TEMP_DIR
                    rm -rf qcore
                    git clone https://github.com/ucgmsim/qcore.git
                    cd qcore
                    python setup.py develop --no-data --no-deps
                """
            }
        }

        stage('Run regression tests') {
            steps {
                echo '[[ Run pytest ]]'
                sh """
# activate virtual environment again
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
# Install may cause the storage going overflow
                    echo "[ Installing ${env.JOB_NAME} ]"
                    python setup.py install
                    echo "[ Linking test data ]"
                    cd empirical/test
                    rm -rf sample0
                    mkdir sample0
                    ln -s /home/qcadmin/data/testing/${env.JOB_NAME}/sample0/input sample0
                    ln -s /home/qcadmin/data/testing/${env.JOB_NAME}/sample0/output sample0
                    echo "[ Run test now ]"
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
                    rm -rf $TEMP_DIR
                """
            }
    }
}
