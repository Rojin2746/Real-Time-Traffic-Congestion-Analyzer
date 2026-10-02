import streamlit as st
import pandas as pd
from datetime import datetime

from src.congestion_analyzer import analyze_video

# ==========================================
# SESSION HISTORY
# ==========================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Traffic Congestion Analyzer",
    page_icon="🚦",
    layout="wide"
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🚦 Traffic Analyzer")

    st.divider()

    st.subheader("📌 Project Information")

    st.write(
        "AI-based traffic monitoring system "
        "using YOLO, OpenCV and Streamlit."
    )

    st.divider()

    st.subheader("🤖 AI Components")

    st.write("• YOLO vehicle detection")
    st.write("• Vehicle tracking")
    st.write("• Speed estimation")
    st.write("• Congestion analysis")
    st.write("• Traffic alerts")

    st.divider()

    st.subheader("📊 Analysis")

    st.write(
        "Upload a traffic video and "
        "start the AI analysis."
    )

    st.divider()

    st.caption(
        "Smart Traffic Monitoring System"
    )

# ==========================================
# TITLE
# ==========================================

st.title("🚦 Real-Time Traffic Congestion Analyzer")

st.caption(
    "AI-powered vehicle detection, tracking, "
    "speed estimation and congestion analysis"
)

st.write(
    "AI-based traffic monitoring using YOLO, "
    "OpenCV and Streamlit."
)

st.divider()


# ==========================================
# VIDEO INPUT
# ==========================================

st.header("📤 Traffic Video Analysis")

uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov"],
    help="Upload an MP4, AVI or MOV traffic video."
)


if uploaded_file is not None:

    # Save uploaded video
    video_path = "data/input/uploaded_traffic.mp4"

    with open(video_path, "wb") as file:
        file.write(uploaded_file.read())

    st.success("Traffic video uploaded successfully.")


    # ==========================================
    # ORIGINAL VIDEO
    # ==========================================

    st.subheader("🎥 Uploaded Traffic Video")

    st.video(video_path)


    # ==========================================
    # START ANALYSIS
    # ==========================================

    if st.button("🚦 Start Traffic Analysis"):

    # ==========================================
    # PROGRESS BAR
    # ==========================================

        progress_bar = st.progress(0)

        status_text = st.empty()


        # ==========================================
        # PROGRESS CALLBACK
        # ==========================================

        def update_progress(
            progress,
            current_frame,
            total_frames
        ):
            

            percentage = int(
                progress * 100
            )

            progress_bar.progress(
                percentage
            )

            status_text.info(
                f"🔄 Processing frame "
                f"{current_frame} / {total_frames} "
                f"({percentage}%)"
            )
        # ==========================================
        # LIVE DASHBOARD PLACEHOLDERS
        # ==========================================

        st.subheader("📡 Live Traffic Monitoring")

        live_col1, live_col2, live_col3, live_col4 = st.columns(4)

        live_total = live_col1.empty()
        live_speed = live_col2.empty()
        live_score = live_col3.empty()
        live_level = live_col4.empty()

        live_alert = st.empty()

        # ==========================================
        # LIVE CHART PLACEHOLDERS
        # ==========================================

        st.subheader("📈 Live Traffic Charts")

        vehicle_chart = st.empty()
        congestion_chart = st.empty()

        live_vehicle_history = []
        live_congestion_history = []

        # ==========================================
        # LIVE DATA CALLBACK
        # ==========================================

        def update_live_data(data):

            live_total.metric(
                "🚘 Vehicles",
                data["total"]
            )

            live_speed.metric(
                "⚡ Speed",
                f"{data['average_speed']:.1f}"
            )

            live_score.metric(
                "📊 Congestion Score",
                f"{data['congestion_score']:.2f}"
            )

            live_level.metric(
                "🚦 Traffic Level",
                data["congestion"]
            )


            if data["congestion"] == "LOW":

                live_alert.success(
                    "🟢 NORMAL TRAFFIC — "
                    "Traffic is flowing normally."
                )

            elif data["congestion"] == "MEDIUM":

                live_alert.warning(
                    "🟡 MODERATE TRAFFIC — "
                    "Traffic is moderately congested."
                )

            elif data["congestion"] == "HIGH":

                live_alert.warning(
                    "🟠 HIGH TRAFFIC — "
                    "High traffic congestion detected."
                )

            else:

                live_alert.error(
                    "🔴 SEVERE TRAFFIC — "
                    "Severe traffic congestion detected."
                )

            # ==========================================
            # UPDATE LIVE CHART DATA
            # ==========================================

            live_vehicle_history.append({
                "frame": data["frame"],
                "vehicles": data["total"]
            })

            live_congestion_history.append({
                "frame": data["frame"],
                "congestion_score": data["congestion_score"]
            })


            # ==========================================
            # VEHICLE COUNT CHART
            # ==========================================

            vehicle_df = pd.DataFrame(
                live_vehicle_history
            )

            vehicle_df = vehicle_df.set_index(
                "frame"
            )

            vehicle_chart.line_chart(
                vehicle_df["vehicles"]
            )


            # ==========================================
            # CONGESTION SCORE CHART
            # ==========================================

            congestion_df = pd.DataFrame(
                live_congestion_history
            )

            congestion_df = congestion_df.set_index(
                "frame"
            )

            congestion_chart.line_chart(
                congestion_df["congestion_score"]
            )


        status_text.info(
            "🔄 Starting traffic analysis..."
        )


        # ==========================================
        # ANALYZE VIDEO
        # ==========================================

        output_path = (
            "data/output/analyzed_traffic.mp4"
        )

        with st.spinner(
            "🤖 AI is analyzing the traffic video..."
        ):

            data = analyze_video(
                video_path,
                output_path,
                max_frames=300,
                progress_callback=update_progress,
                data_callback=update_live_data
            )
        st.header("📊 Traffic Analysis Results")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric(
                "🚘 Total Vehicles",
                data["total"]
            )

        with summary_col2:
            st.metric(
                "⚡ Average Speed",
                f"{data['average_speed']:.1f}"
            )

        with summary_col3:
            st.metric(
                "📊 Congestion Score",
                f"{data['congestion_score']:.2f}"
            )

        with summary_col4:
            st.metric(
                "🚦 Congestion Level",
                data["congestion"]
            )

        st.divider()

        # ==========================================
        # SAVE ANALYSIS RESULT
        # ==========================================

        st.session_state.analysis_history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Vehicles": data["total"],
            "Cars": data["cars"],
            "Motorcycles": data["motorcycles"],
            "Buses": data["buses"],
            "Trucks": data["trucks"],
            "Average Speed": round(data["average_speed"], 2),
            "Congestion Score": round(
                data["congestion_score"],
                2
            ),
            "Congestion": data["congestion"]
        })
        # ==========================================
        # COMPLETED
        # ==========================================

        progress_bar.progress(100)

        status_text.success(
            "✅ Traffic analysis completed!"
        )

        # ==========================================
        # CONGESTION GAUGE
        # ==========================================

        st.subheader("🚦 Congestion Level")

        congestion_score = data["congestion_score"]

        gauge_percentage = int(
            congestion_score * 100
        )

        st.progress(
            gauge_percentage
        )

        st.write(
            f"Congestion Score: "
            f"**{congestion_score:.2f} / 1.00**"
        )

        st.write(
            f"Current Level: "
            f"**{data['congestion']}**"
        )

        # ==========================================
        # DASHBOARD SUMMARY CARDS
        # ==========================================

        st.subheader("📊 Traffic Dashboard Summary")

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🚘 Total Vehicles",
                data["total"]
            )


        with col2:

            st.metric(
                "🚗 Cars",
                data["cars"]
            )


        with col3:

            st.metric(
                "🏍 Motorcycles",
                data["motorcycles"]
            )


        with col4:

            st.metric(
                "🚌 Buses",
                data["buses"]
            )

        # ==========================================
        # CONGESTION GAUGE
        # ==========================================

        st.subheader("🎯 Congestion Level Gauge")

        congestion_score = data["congestion_score"]

        gauge_percentage = congestion_score * 100

        st.progress(
            int(gauge_percentage)
        )

        st.write(
            f"Congestion: **{gauge_percentage:.1f}%**"
        )

        st.caption(
            "0% = Very Low Traffic | "
            "100% = Severe Traffic"
        )

        # ==========================================
        # TRAFFIC ANALYSIS REPORT
        # ==========================================

        st.subheader("📄 Traffic Analysis Report")

        report_text = f"""
        REAL-TIME TRAFFIC CONGESTION ANALYZER
        =====================================

        Analysis Date:
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        VEHICLE STATISTICS
        ------------------
        Cars: {data["cars"]}
        Motorcycles: {data["motorcycles"]}
        Buses: {data["buses"]}
        Trucks: {data["trucks"]}
        Total Vehicles: {data["total"]}

        TRAFFIC ANALYSIS
        ----------------
        Average Speed: {data["average_speed"]:.2f}
        Congestion Score: {data["congestion_score"]:.2f}
        Congestion Percentage: {data["congestion_score"] * 100:.1f}%
        Congestion Level: {data["congestion"]}

        TRAFFIC ALERT
        -------------
        {data["alert"]["title"]}
        {data["alert"]["message"]}
        """

        st.download_button(
            label="Download Traffic Report",
            data=report_text,
            file_name="traffic_analysis_report.txt",
            mime="text/plain"
        )


        col5, col6, col7, col8 = st.columns(4)


        with col5:

            st.metric(
                "🚚 Trucks",
                data["trucks"]
            )


        with col6:

            st.metric(
                "⚡ Average Speed",
                f"{data['average_speed']:.1f}"
            )


        with col7:

            st.metric(
                "📊 Congestion Score",
                f"{data['congestion_score']:.2f}"
            )


        with col8:

            st.metric(
                "🚦 Traffic Level",
                data["congestion"]
            )

        # ==========================================
        # TRAFFIC ALERT
        # ==========================================

        st.subheader("🚨 Traffic Alert")

        alert = data["alert"]


        if alert["type"] == "success":

            st.success(
                f"{alert['title']}\n\n"
                f"{alert['message']}"
            )


        elif alert["type"] == "warning":

            st.warning(
                f"{alert['title']}\n\n"
                f"{alert['message']}"
            )


        else:

            st.error(
                f"{alert['title']}\n\n"
                f"{alert['message']}"
            )
        # ==========================================
        # AI PROCESSED VIDEO
        # ==========================================

        st.subheader(
            " AI Processed Traffic Video"
        )

        st.video(output_path)


        # ==========================================
        # CREATE DATAFRAME
        # ==========================================

        history = data["vehicle_history"]

        df = pd.DataFrame(history)

        # ==========================================
        # HISTORY CHARTS
        # ==========================================

        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:

            st.subheader("📈 Vehicle Count History")

            if not df.empty:

                chart_data = df.set_index("frame")

                st.line_chart(
                    chart_data["total"]
                )

            else:

                st.info(
                    "No vehicle history available."
                )


        with chart_col2:

            st.subheader("🚦 Congestion History")

            if not df.empty:

                congestion_data = df.set_index(
                    "frame"
                )

                st.line_chart(
                    congestion_data["congestion_score"]
                )

            else:

                st.info(
                    "No congestion history available."
                )
        

        # ==========================================
        # VEHICLE ANALYSIS
        # ==========================================

        st.header("🚗 Vehicle Analysis")

        vehicle_col1, vehicle_col2 = st.columns(2)


        with vehicle_col1:

            st.subheader("Vehicle Type Comparison")

            if not df.empty:

                vehicle_type_data = df[
                    [
                        "cars",
                        "motorcycles",
                        "buses",
                        "trucks"
                    ]
                ].mean()

                st.bar_chart(
                    vehicle_type_data
                )

            else:

                st.info(
                    "No vehicle type data available."
                )


        with vehicle_col2:

            st.subheader("Vehicle Counts")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🚗 Cars",
                    data["cars"]
                )

                st.metric(
                    "🚌 Buses",
                    data["buses"]
                )

            with col2:

                st.metric(
                    "🏍 Motorcycles",
                    data["motorcycles"]
                )

                st.metric(
                    "🚚 Trucks",
                    data["trucks"]
                )


        # ==========================================
        # TRAFFIC CONDITION MESSAGE
        # ==========================================

        st.subheader(
            "Traffic Condition"
        )


        if data["congestion"] == "LOW":

            st.success(
                "Traffic condition is currently LOW."
            )


        elif data["congestion"] == "MEDIUM":

            st.warning(
                "Traffic condition is currently MEDIUM."
            )


        elif data["congestion"] == "HIGH":

            st.warning(
                "Traffic condition is currently HIGH."
            )


        else:

            st.error(
                "Traffic condition is currently SEVERE."
            )


else:

    st.info(
        "Upload a traffic video to start analysis."
    )

# ==========================================
# ANALYSIS SESSION HISTORY
# ==========================================

st.subheader("Analysis Session History")

if st.session_state.analysis_history:

    history_df = pd.DataFrame(
        st.session_state.analysis_history
    )

    # Display history table
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================

    csv_data = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Session History (CSV)",
        data=csv_data,
        file_name="traffic_analysis_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No analysis sessions recorded yet."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Real-Time Traffic Congestion Analyzer | "
    "YOLO + OpenCV + Streamlit"
)
