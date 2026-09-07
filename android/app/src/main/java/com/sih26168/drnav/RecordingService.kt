package com.sih26168.drnav

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.location.Location
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

data class RecorderState(
    val recording: Boolean = false,
    val elapsedS: Double = 0.0,
    val imuRows: Long = 0,
    val gnssRows: Long = 0,
    val sky: SkyState = SkyState(),
    val dir: String? = null,
    val secondsSinceFix: Double = Double.NaN,
    val message: String = "",
)

/**
 * Foreground service that records one drive.
 *
 * It runs in the foreground for a boring but decisive reason: the phone will be
 * in a pocket or a cradle with the screen off for the interesting part of the
 * recording, and a backgrounded process loses sensor callbacks the moment
 * Android decides to doze. A dropped minute of IMU in the middle of a tunnel
 * ruins the only run you drove out to collect.
 */
class RecordingService : Service() {

    companion object {
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 1
        const val ACTION_START = "com.sih26168.drnav.START"
        const val ACTION_STOP = "com.sih26168.drnav.STOP"
        const val EXTRA_MODE = "mode"

        private val _state = MutableStateFlow(RecorderState())
        val state: StateFlow<RecorderState> = _state.asStateFlow()
    }

    private lateinit var thread: HandlerThread
    private lateinit var handler: Handler
    private var imu: ImuCollector? = null
    private var gnss: GnssCollector? = null
    private var writer: LogWriter? = null

    private var t0Ns = 0L
    private var origin: DoubleArray? = null      // lat, lon, alt of the first fix
    private var lastFixElapsedNs = 0L
    private var mode = "vehicle"

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopRecording(); stopSelf(); return START_NOT_STICKY }
            else -> {
                mode = intent?.getStringExtra(EXTRA_MODE) ?: "vehicle"
                startRecording()
            }
        }
        return START_STICKY
    }

    private fun startRecording() {
        if (_state.value.recording) return

        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(getExternalFilesDir(null), "recordings/$mode-$stamp")
        writer = LogWriter(dir)

        t0Ns = SystemClock.elapsedRealtimeNanos()
        lastFixElapsedNs = t0Ns

        thread = HandlerThread("recorder").apply { start() }
        handler = Handler(thread.looper)

        val imuRate = 200
        imu = ImuCollector(this, imuRate, ::onImuSample, ::onBaroSample).also {
            if (!it.available) {
                _state.value = _state.value.copy(message = "no accelerometer or gyroscope")
            }
            it.start(handler)
        }
        gnss = GnssCollector(this, ::onFix, ::onSky, { event ->
            val t = elapsed(SystemClock.elapsedRealtimeNanos())
            event.measurements.forEach { m ->
                writer?.gnssRaw(
                    t = t,
                    svid = m.svid,
                    constellation = m.constellationType,
                    cn0 = m.cn0DbHz,
                    prRate = m.pseudorangeRateMetersPerSecond,
                    prRateUnc = m.pseudorangeRateUncertaintyMetersPerSecond,
                    carrierHz = if (m.hasCarrierFrequencyHz()) m.carrierFrequencyHz.toDouble() else 0.0,
                    multipath = m.multipathIndicator,
                    state = m.state,
                )
            }
        }).also { it.start(handler) }

        startForeground(NOTIFICATION_ID, buildNotification("Recording $mode"))
        _state.value = RecorderState(recording = true, dir = dir.absolutePath,
                                     message = "waiting for first GNSS fix")
    }

    private fun stopRecording() {
        if (!_state.value.recording) return
        imu?.stop(); gnss?.stop()
        origin?.let { o ->
            writer?.writeMeta(mode, o[0], o[1], o[2], 200.0,
                              notes = "recorded by DR Logger")
        }
        writer?.close()
        if (::thread.isInitialized) thread.quitSafely()
        _state.value = _state.value.copy(
            recording = false,
            message = if (origin == null) "stopped with no GNSS fix -- meta.json not written"
                      else "saved",
        )
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    override fun onDestroy() { stopRecording(); super.onDestroy() }

    // ---- callbacks -------------------------------------------------------

    private fun elapsed(ns: Long) = (ns - t0Ns) / 1e9

    private fun onImuSample(tNs: Long, accel: FloatArray, gyro: FloatArray) {
        writer?.imu(elapsed(tNs), accel, gyro)
        val w = writer ?: return
        if (w.imuRows % 200L == 0L) publish()
    }

    private fun onBaroSample(tNs: Long, hPa: Float) = writer?.baro(elapsed(tNs), hPa) ?: Unit

    private fun onFix(loc: Location) {
        val o = origin ?: doubleArrayOf(loc.latitude, loc.longitude, loc.altitude)
            .also { origin = it }
        val enu = Enu.fromGeodetic(loc.latitude, loc.longitude, loc.altitude, o[0], o[1], o[2])

        // Android reports horizontal motion as speed plus bearing rather than a
        // velocity vector. Bearing is degrees clockwise from true north, which
        // matches the heading convention used throughout the offline pipeline.
        val bearing = Math.toRadians(loc.bearing.toDouble())
        val speed = loc.speed.toDouble()
        val sky = _state.value.sky

        lastFixElapsedNs = loc.elapsedRealtimeNanos
        writer?.gnss(
            t = elapsed(loc.elapsedRealtimeNanos),
            e = enu[0], n = enu[1], u = enu[2],
            ve = speed * sin(bearing), vn = speed * cos(bearing), vu = 0.0,
            nsat = sky.satellitesInFix,
            cn0 = sky.meanCn0,
            hdop = if (sky.hdop.isNaN()) 1.0 else sky.hdop,
            acc = loc.accuracy.toDouble(),
        )
        publish()
    }

    private fun onSky(sky: SkyState) { _state.value = _state.value.copy(sky = sky) }

    private fun publish() {
        val now = SystemClock.elapsedRealtimeNanos()
        _state.value = _state.value.copy(
            elapsedS = elapsed(now),
            imuRows = writer?.imuRows ?: 0,
            gnssRows = writer?.gnssRows ?: 0,
            secondsSinceFix = (now - lastFixElapsedNs) / 1e9,
            message = if (origin == null) "waiting for first GNSS fix" else "recording",
        )
    }

    // ---- notification ----------------------------------------------------

    private fun buildNotification(text: String): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, getString(R.string.channel_name),
                                    NotificationManager.IMPORTANCE_LOW))
        }
        val tap = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(tap)
            .setOngoing(true)
            .build()
    }
}
