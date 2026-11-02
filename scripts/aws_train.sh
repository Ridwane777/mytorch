#!/bin/sh
# Trains a configuration on a temporary EC2 instance, unattended: you can switch your computer off
# after "launch". Everything runs in the selected Region of the AWS profile (ADR-0038).
#   sh scripts/aws_train.sh launch [CONFIG]   uploads the code and the dataset to a private S3
#                                             bucket, starts one on-demand instance, which trains
#                                             with scripts/train_final.sh, uploads runs/<name>/ to
#                                             S3 every 10 minutes (progress.log, the weights
#                                             every PROGRESS_EVERY epochs, the checkpoint) and at
#                                             the end, then terminates itself;
#   sh scripts/aws_train.sh status            instance state and last progress lines;
#   sh scripts/aws_train.sh fetch             downloads the results into runs/aws/<name>/;
#   sh scripts/aws_train.sh cleanup           terminates the instance if any, empties and deletes
#                                             the bucket and the IAM role (costs stop).
# Environment: AWS_PROFILE (default azan), INSTANCE_TYPE (default c7i.8xlarge: refused on the
# free plan, which accepts only small types such as c7i-flex.large), EPOCHS and PATIENCE
# (override the epochs= and patience= of the configuration), PROGRESS_EVERY (default 5), FRACTION (default 1: the whole
# dataset), DISK_GB (default 40), MAX_MINUTES (default: effective epochs x 40 + 60, see below),
# RESUME_CHECKPOINT (a checkpoint.ckpt downloaded from an earlier run, uploaded to the new bucket
# so that the instance resumes from it).
#
# Epochs: a snapshot is written every PROGRESS_EVERY epochs, so the effective number of epochs
# (EPOCHS if set, else the epochs= of the configuration) must be at least PROGRESS_EVERY: launch
# refuses otherwise, before any call to AWS (configs/cnn_network.conf has epochs=30: use for
# instance EPOCHS=200, or a lower PROGRESS_EVERY). The early stopping of the configuration
# (patience=) can still end the training before the first snapshot; the final network.nn and
# the checkpoint are always uploaded. To use PROGRESS_EVERY=50 with configs/cnn_network.conf and
# run at least 50 epochs: PROGRESS_EVERY=50 EPOCHS=200 PATIENCE=0 sh scripts/aws_train.sh launch. PATIENCE=0 turns
# the early stopping off, and then the best weights are NOT restored: network.nn is the last epoch,
# and the snapshots network.nn.<date>-<time>.epochN.nn (and curve.png) are the way to pick an earlier epoch.
#
# No Spot: a reclaimed instance would stop the training. There is no automatic resume either: if
# a run fails, run "fetch" (it downloads checkpoint.ckpt), "cleanup", then launch again with the
# same EPOCHS, PATIENCE, FRACTION and CONFIG and RESUME_CHECKPOINT=runs/aws/<name>/checkpoint.ckpt.
# The instance terminates itself (shutdown behavior "terminate"); if that fails the bill continues
# until "cleanup": check "status" after the expected duration.
# MAX_MINUTES is a safety net against a hang (snap, apt, S3, training): the instance schedules
# "shutdown -h +MAX_MINUTES" as soon as it boots. It can also cut a legitimately long run: the
# final upload then does not happen, and what survives is what the uploads every 10 minutes
# already sent (progress.log, snapshots, checkpoint.ckpt). Raise MAX_MINUTES for a long run.
set -eu

cd "$(dirname "$0")/.."
export AWS_PROFILE="${AWS_PROFILE:-azan}"
export AWS_PAGER=""
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.8xlarge}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
FRACTION="${FRACTION:-1}"
DISK_GB="${DISK_GB:-40}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MAX_MINUTES="${MAX_MINUTES:-}"
STATE=".aws_train_state"

fail() {
    echo "aws_train.sh: $1" >&2
    exit 1
}
need_state() {
    [ -f "$STATE" ] || fail "no run launched from this directory (no $STATE)"
    # shellcheck disable=SC1090
    . "./$STATE"
}
write_state() {
    cat > "$STATE" <<EOF
BUCKET=$BUCKET
ROLE=$ROLE
INSTANCE=$INSTANCE
REGION=$REGION
NAME=$NAME
EOF
}

cmd_launch() {
    CONFIG="${1:-configs/cnn_network.conf}"
    [ -f "$CONFIG" ] || fail "$CONFIG not found"
    [ ! -f "$STATE" ] || fail "a run is already recorded in $STATE: fetch / cleanup first"
    NAME="$(basename "$CONFIG" .conf)"

    # 0. Checks that need no call to AWS: the snapshots must be reachable.
    case "$EPOCHS$PATIENCE$PROGRESS_EVERY$MAX_MINUTES" in
    *[!0-9]*) fail "EPOCHS, PATIENCE (0 = no early stopping), PROGRESS_EVERY and MAX_MINUTES must be integers" ;;
    esac
    [ -z "$MAX_MINUTES" ] || [ "$MAX_MINUTES" -ge 1 ] || fail "MAX_MINUTES must be at least 1"
    [ "$PROGRESS_EVERY" -ge 1 ] || fail "PROGRESS_EVERY must be at least 1"
    [ -z "$EPOCHS" ] || [ "$EPOCHS" -ge 1 ] || fail "EPOCHS must be at least 1"
    config_epochs="$(sed -n 's/.*[[:space:]]epochs=\([0-9][0-9]*\).*/\1/p' "$CONFIG" | head -n 1)"
    EFFECTIVE_EPOCHS="${EPOCHS:-$config_epochs}"
    [ -n "$EFFECTIVE_EPOCHS" ] || fail "no epochs= in $CONFIG: set EPOCHS"
    if [ "$EFFECTIVE_EPOCHS" -lt "$PROGRESS_EVERY" ]; then
        fail "only $EFFECTIVE_EPOCHS epochs would run, less than PROGRESS_EVERY=$PROGRESS_EVERY: no snapshot would be written. Set EPOCHS (for instance EPOCHS=200) or lower PROGRESS_EVERY."
    fi
    MAX_MINUTES="${MAX_MINUTES:-$((EFFECTIVE_EPOCHS * 40 + 60))}"
    config_patience="$(sed -n 's/.*[[:space:]]patience=\([0-9][0-9]*\).*/\1/p' "$CONFIG" | head -n 1)"
    EFFECTIVE_PATIENCE="${PATIENCE:-${config_patience:-0}}"
    if [ "$EFFECTIVE_PATIENCE" -gt 0 ]; then
        echo "note: early stopping (patience=$EFFECTIVE_PATIENCE): the training can end before epoch" \
            "$PROGRESS_EVERY; the final network.nn and the checkpoint are always uploaded" >&2
    else
        echo "note: no early stopping (patience=0): the best weights are not restored, network.nn is" \
            "the last epoch; the snapshots are the safety net" >&2
    fi
    [ -z "$RESUME_CHECKPOINT" ] || [ -f "$RESUME_CHECKPOINT" ] ||
        fail "RESUME_CHECKPOINT $RESUME_CHECKPOINT not found"

    REGION="$(aws configure get region || true)"
    [ -n "$REGION" ] || fail "no Region for profile $AWS_PROFILE; run: aws login --profile $AWS_PROFILE"
    ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" ||
        fail "not signed in; run: aws login --profile $AWS_PROFILE"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    BUCKET="my-torch-$ACCOUNT-$REGION-$STAMP"
    ROLE="my-torch-train-$STAMP"
    # The instance tag, and the value that the policy of the role below must match.
    TAG="my-torch-$STAMP"
    INSTANCE=""
    echo "Region $REGION, bucket $BUCKET, instance $INSTANCE_TYPE"
    echo "EPOCHS=$EFFECTIVE_EPOCHS PATIENCE=$EFFECTIVE_PATIENCE PROGRESS_EVERY=$PROGRESS_EVERY MAX_MINUTES=$MAX_MINUTES"
    # Recorded before anything is created, so that "cleanup" also works after a failed launch.
    write_state
    TMP="$(mktemp -d)"
    trap 'rc=$?; rm -rf "$TMP"; [ "$rc" -eq 0 ] || echo "aws_train.sh: launch failed; run: sh scripts/aws_train.sh cleanup" >&2' EXIT

    # 1. Private bucket (encrypted by default); code and dataset go in as one archive. The
    # dataset is part of it; build products, previous runs, snapshots, the AWS CLI installer
    # and the student resources are not.
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration "LocationConstraint=$REGION" > /dev/null
    aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    tar --exclude='./build' --exclude='./runs' --exclude='./aws' --exclude='*.zip' \
        --exclude='./student_resources_MyTorch' --exclude='*.gcda' --exclude='*.gcno' \
        --exclude='./.git' --exclude="./$STATE" --exclude='*.epoch*.nn' \
        --exclude='./my_torch_analyzer' --exclude='./my_torch_generator' \
        --exclude='./unit_tests*' -czf "$TMP/code.tar.gz" .
    aws s3 cp "$TMP/code.tar.gz" "s3://$BUCKET/code.tar.gz" --only-show-errors
    if [ -n "$RESUME_CHECKPOINT" ]; then
        aws s3 cp "$RESUME_CHECKPOINT" "s3://$BUCKET/runs/$NAME/checkpoint.ckpt" --only-show-errors
    fi

    # 2. Instance role: read/write on this bucket, terminate the instances tagged for this run,
    # and Session Manager to look inside if needed.
    cat > "$TMP/trust.json" <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
    cat > "$TMP/policy.json" <<EOF
{"Version":"2012-10-17","Statement":[
{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
{"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$BUCKET"},
{"Effect":"Allow","Action":"ec2:TerminateInstances","Resource":"*",
 "Condition":{"StringEquals":{"aws:ResourceTag/Project":"$TAG"}}}]}
EOF
    aws iam create-role --role-name "$ROLE" --assume-role-policy-document "file://$TMP/trust.json" > /dev/null
    aws iam put-role-policy --role-name "$ROLE" --policy-name run --policy-document "file://$TMP/policy.json"
    aws iam attach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    aws iam create-instance-profile --instance-profile-name "$ROLE" > /dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
    sleep 15 # IAM propagation

    # 3. User data (bash, run as root by cloud-init): install, build, train, upload, terminate.
    # The heredoc is not quoted: the values known now ($BUCKET, $NAME, $REGION, $CONFIG, $FRACTION,
    # $PROGRESS_EVERY, $EPOCHS, $PATIENCE, $MAX_MINUTES) are expanded here, every runtime variable
    # and command substitution is escaped (\$).
    cat > "$TMP/userdata.sh" <<EOF
#!/bin/bash
export HOME=/root DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION=$REGION
export PATH="\$PATH:/snap/bin"
BUCKET=$BUCKET
RUN=runs/$NAME
exec > /var/log/train.log 2>&1
# Safety net against a hang: the shutdown behavior is "terminate" (see MAX_MINUTES in aws_train.sh).
shutdown -h +$MAX_MINUTES

upload() {
    aws s3 sync "/work/\$RUN" "s3://\$BUCKET/\$RUN" --only-show-errors --exclude '*' --include 'network.nn' --include 'network.nn.*.epoch*.nn' --include 'checkpoint.ckpt' --include 'epochs.log' --include 'epochs/*' --include 'progress.log' --include 'training.log' --include 'run.txt' --include 'scores.txt' --include 'predictions.txt' --include 'curve.png' --include 'confusion.png'
    aws s3 cp /var/log/train.log "s3://\$BUCKET/train.log" --only-show-errors
}

# Runs on every exit, success or failure: upload what exists, then make the instance disappear.
# The shutdown behavior is "terminate", so "shutdown" is the fallback when terminate-instances
# does not work (no CLI, no permission, no network...).
finish() {
    rc=\$?
    set +e
    trap - EXIT
    [ -z "\$LOOP" ] || kill "\$LOOP" 2> /dev/null
    upload || echo "upload failed"
    echo "exit status \$rc" > /tmp/done
    aws s3 cp /tmp/done "s3://\$BUCKET/DONE" --only-show-errors || echo "DONE marker not uploaded"
    token=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
    instance=\$(curl -s -H "X-aws-ec2-metadata-token: \$token" http://169.254.169.254/latest/meta-data/instance-id)
    if [ -z "\$instance" ] || ! aws ec2 terminate-instances --instance-ids "\$instance" > /dev/null; then
        echo "terminate-instances failed: shutting down"
        shutdown -h now
    fi
}
LOOP=""
trap finish EXIT

set -euo pipefail
snap wait system seed.loaded || true
command -v aws > /dev/null || snap install aws-cli --classic
apt-get update -y
apt-get install -y g++ make git time unzip python3-matplotlib
mkdir -p /work
cd /work
aws s3 cp "s3://\$BUCKET/code.tar.gz" - | tar --no-same-owner -xzf -
mkdir -p "\$RUN"
aws s3 cp "s3://\$BUCKET/\$RUN/checkpoint.ckpt" "\$RUN/checkpoint.ckpt" --only-show-errors || true
# train_final.sh writes the commit in run.txt: a repository of the uploaded code (the dataset is
# ignored by .gitignore).
git config --global --add safe.directory "*"
git init -q .
git add -A
git -c user.name=train -c user.email=train@localhost commit -qm "code uploaded by aws_train.sh"
( while sleep 600; do upload || true; done ) &
LOOP=\$!
FRACTION=$FRACTION PROGRESS_EVERY=$PROGRESS_EVERY EPOCHS=$EPOCHS PATIENCE=$PATIENCE sh scripts/train_final.sh "$CONFIG" "\$RUN"
EOF

    # 4. Launch: Ubuntu 24.04, encrypted gp3, IMDSv2 required, no key pair, no inbound rule.
    AMI="$(aws ssm get-parameter --name \
        /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
        --query Parameter.Value --output text)"
    # The Project tag below is the value that the policy condition above requires.
    INSTANCE="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --user-data "file://$TMP/userdata.sh" \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
        --block-device-mappings \
        "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$TAG}]" \
        --query 'Instances[0].InstanceId' --output text)"
    write_state
    echo "Launched $INSTANCE. You can switch your computer off."
    echo "Later: sh scripts/aws_train.sh status | fetch | cleanup"
}

cmd_status() {
    need_state
    if [ -n "$INSTANCE" ]; then
        state="$(aws ec2 describe-instances --instance-ids "$INSTANCE" \
            --query 'Reservations[0].Instances[0].State.Name' --output text 2> /dev/null || echo unknown)"
    else
        state="none (launch did not reach the instance)"
    fi
    echo "instance ${INSTANCE:--}: $state"
    if done_marker="$(aws s3 cp "s3://$BUCKET/DONE" - 2> /dev/null)"; then
        echo "finished ($done_marker): run fetch"
    else
        echo "not finished"
    fi
    aws s3 cp "s3://$BUCKET/runs/$NAME/progress.log" - 2> /dev/null | tail -n 5 || true
}

cmd_fetch() {
    need_state
    mkdir -p "runs/aws/$NAME"
    aws s3 sync "s3://$BUCKET/runs/$NAME" "runs/aws/$NAME" --only-show-errors
    aws s3 cp "s3://$BUCKET/train.log" "runs/aws/$NAME/train.log" --only-show-errors || true
    echo "Results in runs/aws/$NAME/ (network.nn, network.nn.<date>-<time>.epochN.nn, checkpoint.ckpt, progress.log, run.txt...)"
    echo "Then: sh scripts/aws_train.sh cleanup"
}

cmd_cleanup() {
    need_state
    aws sts get-caller-identity > /dev/null ||
        fail "not signed in; run: aws login --profile $AWS_PROFILE (state kept)"
    if [ -n "$INSTANCE" ]; then
        aws ec2 terminate-instances --instance-ids "$INSTANCE" > /dev/null 2>&1 || true
        # The instance uploads once more when it stops: wait, or that upload could land between
        # the emptying and the deletion of the bucket.
        aws ec2 wait instance-terminated --instance-ids "$INSTANCE" || true
    fi
    aws s3 rm "s3://$BUCKET" --recursive --only-show-errors || true
    aws s3api delete-bucket --bucket "$BUCKET" || true
    aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" || true
    aws iam delete-instance-profile --instance-profile-name "$ROLE" || true
    aws iam detach-role-policy --role-name "$ROLE" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore || true
    aws iam delete-role-policy --role-name "$ROLE" --policy-name run || true
    aws iam delete-role --role-name "$ROLE" || true
    if aws s3api head-bucket --bucket "$BUCKET" 2> /dev/null ||
        aws iam get-role --role-name "$ROLE" > /dev/null 2>&1; then
        fail "some resources remain; state kept in $STATE, run cleanup again"
    fi
    rm -f "$STATE"
    echo "Cleaned up."
}

case "${1:-}" in
launch) shift; cmd_launch "$@" ;;
status) cmd_status ;;
fetch) cmd_fetch ;;
cleanup) cmd_cleanup ;;
*) fail "usage: sh scripts/aws_train.sh launch [CONFIG] | status | fetch | cleanup" ;;
esac
